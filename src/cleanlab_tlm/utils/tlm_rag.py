"""
A version of the Trustworthy Language Model, specifically for real-time validation of RAG responses (Trust Augmented Generation, or TAG).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Coroutine, Sequence
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, cast

from tqdm.asyncio import tqdm_asyncio
from typing_extensions import (  # for Python <3.11 with (Not)Required
    NotRequired,
    TypedDict,
)

from cleanlab_tlm.errors import (
    MissingApiKeyError,
    ValidationError,
)
from cleanlab_tlm.internal.api.api import tlm_rag_generate, tlm_rag_score
from cleanlab_tlm.internal.concurrency import TlmRateHandler
from cleanlab_tlm.internal.exception_handling import handle_tlm_exceptions
from cleanlab_tlm.tlm import TLMOptions, is_notebook

if TYPE_CHECKING:
    from cleanlab_tlm.internal.types import TLMQualityPreset


class TrustworthyRAG:
    def __init__(
        self,
        quality_preset: TLMQualityPreset = "medium",
        *,
        api_key: Optional[str] = None,
        options: Optional[TLMOptions] = None,
        timeout: Optional[float] = None,
        verbose: Optional[bool] = None,
        evals: list[Eval] = None,
    ) -> None:
        """
        lazydocs: ignore
        """
        self._api_key = api_key or os.environ.get("CLEANLAB_TLM_API_KEY")
        if self._api_key is None:
            raise MissingApiKeyError

        # Validate quality_preset
        if quality_preset not in ["medium", "low"]:
            raise ValidationError(f"Unsupported quality_preset: '{quality_preset}'. Only 'medium' and 'low' are supported.")

        self._quality_preset = quality_preset
        self._options = options

        # If evals not provided, use the default evals defined in this file
        if evals is None:
            self._evals = [
                Eval(
                    name=eval_config["name"],
                    criteria=eval_config["criteria"],
                    query_identifier=eval_config.get("query_identifier"),
                    context_identifier=eval_config.get("context_identifier"),
                    response_identifier=eval_config.get("response_identifier"),
                )
                for eval_config in DEFAULT_EVALS
            ]
        else:
            self._evals = evals

        is_notebook_flag = is_notebook()

        self._timeout = timeout if timeout is not None and timeout > 0 else None
        self._verbose = verbose if verbose is not None else is_notebook_flag

        if is_notebook_flag:
            import nest_asyncio  # type: ignore

            nest_asyncio.apply()

        try:
            self._event_loop = asyncio.get_event_loop()
        except RuntimeError:
            self._event_loop = asyncio.new_event_loop()
        self._rate_handler = TlmRateHandler()

    def generate(
        self,
        *,
        query: Union[str, Sequence[str]],
        context: Union[str, Sequence[str]],
        prompt: Optional[Union[str, Sequence[str]]] = None,
        form_prompt: Optional[Callable[[str, str], str]] = None
    ) -> Union[TrustworthyRAGResponse, list[TrustworthyRAGResponse]]:
        """ Required parameters are query and context. Either prompt or form_prompt can be provided optionally.
           default_prompt_format = function that takes in {query, context} 
           returns a standard RAG prompt like we use for Agility.
           
           Args:
                query: Either a single query string or a sequence of query strings for batch processing.
                      This parameter is required.
                context: Either a single context string or a sequence of context strings for batch processing.
                        This parameter is required.
                prompt: Either a single prompt string or a sequence of prompt strings for batch processing.
                       This parameter is optional and mainly used for replacing the default prompt format entirely.
                form_prompt: Optional function to format the prompt. Cannot be provided together with prompt.
                       The function should take query and context as parameters and return a formatted prompt string.
                       Note: If you need to include a system prompt or any other special instructions, you must
                       incorporate them directly in your custom form_prompt function implementation.
                       
           Returns:
                TrustworthyRAGResponse | list[TrustworthyRAGResponse]: A single response or a list of responses
                if batch processing was used. Each response contains the generated text and evaluation scores.
        """
        # Validate that prompt and form_prompt are not provided at the same time
        if prompt is not None and form_prompt is not None:
            raise ValidationError("'prompt' and 'form_prompt' cannot be provided at the same time. Use either one, not both.")

        # Use the class method as default if form_prompt is not provided and prompt is None
        if prompt is None and form_prompt is None:
            form_prompt = self._default_prompt_formatter

        # Check if we're dealing with batch inputs
        is_batch = False
        batch_size = None

        # Check if any of the inputs are sequences (for batch processing)
        if isinstance(query, Sequence) and not isinstance(query, str):
            is_batch = True
            batch_size = len(query)
        elif isinstance(context, Sequence) and not isinstance(context, str):
            is_batch = True
            batch_size = len(context)
        elif isinstance(prompt, Sequence) and not isinstance(prompt, str):
            is_batch = True
            batch_size = len(prompt)

        # Handle single input case
        if not is_batch:
            # Validate that query and context are provided
            if query is None or context is None:
                raise ValidationError("Both 'query' and 'context' are required parameters")

            if prompt is None:
                prompt = form_prompt(query, context)

            # Validate parameter types
            for param_name, param_value in [
                ("prompt", prompt),
                ("query", query),
                ("context", context),
            ]:
                if param_value is not None and not isinstance(param_value, str):
                    raise ValidationError(f"'{param_name}' must be a string")

            # Validate inputs for evaluations (excluding response which isn't available yet)
            is_valid, error_message = self._validate_inputs(
                query=query,
                context=context,
            )
            if not is_valid:
                raise ValidationError(error_message)

            response = self._event_loop.run_until_complete(
                self._generate_async(
                    prompt=prompt,
                    query=query,
                    context=context,
                )
            )

            # Return the response with the updated TrustworthyRAGResponse format
            return TrustworthyRAGResponse(**response)

        # Handle batch input case
        else:
            # Ensure all batch inputs have the same length
            batch_inputs = []
            for param_name, param_value in [
                ("query", query),
                ("context", context),
                ("prompt", prompt),
            ]:
                if param_value is not None:
                    if isinstance(param_value, Sequence) and not isinstance(param_value, str):
                        if batch_size is not None and len(param_value) != batch_size:
                            raise ValidationError(f"All batch inputs must have the same length, but '{param_name}' has length {len(param_value)} while expected {batch_size}")
                        batch_inputs.append((param_name, param_value))
                    elif isinstance(param_value, str):
                        # Convert single string to a list of the same string repeated batch_size times
                        batch_inputs.append((param_name, [param_value] * batch_size))
                    else:
                        raise ValidationError(f"'{param_name}' must be a string or a sequence of strings")

            # Create dictionaries for each batch item
            batch_items = [{} for _ in range(batch_size)]
            for param_name, param_values in batch_inputs:
                for i, value in enumerate(param_values):
                    batch_items[i][param_name] = value

            # Process each batch item
            processed_prompts = []
            processed_queries = []
            processed_contexts = []

            for item in batch_items:
                item_prompt = item.get("prompt")
                item_query = item.get("query")
                item_context = item.get("context")

                # Validate that query and context are provided for each batch item
                if item_query is None or item_context is None:
                    raise ValidationError("Both 'query' and 'context' are required parameters for each batch item")

                # Generate prompt if not provided
                if item_prompt is None:
                    item_prompt = form_prompt(item_query, item_context)

                # Validate inputs for evaluations
                is_valid, error_message = self._validate_inputs(
                    query=item_query,
                    context=item_context,
                )
                if not is_valid:
                    raise ValidationError(error_message)

                processed_prompts.append(item_prompt)
                processed_queries.append(item_query)
                processed_contexts.append(item_context)

            # Run batch processing
            responses = self._event_loop.run_until_complete(
                self._batch_generate(
                    prompts=processed_prompts,
                    queries=processed_queries,
                    contexts=processed_contexts,
                    capture_exceptions=False,
                )
            )

            # Convert responses to TrustworthyRAGResponse format
            return [TrustworthyRAGResponse(**response) for response in responses]

    def score(
        self,
        *,
        response: Union[str, Sequence[str]],
        query: Union[str, Sequence[str]],
        context: Union[str, Sequence[str]],
        prompt: Optional[Union[str, Sequence[str]]] = None,
        form_prompt: Optional[Callable[[str, str], str]] = None
    ) -> Union[TrustworthyRAGScore, list[TrustworthyRAGScore]]:
        """ Required is either specifying at least: query and context, or prompt. 
           default_prompt_format = function that takes in {query, context} 
           returns a standard RAG prompt like we use for Agility.
           
           Args:
                response: Either a single response string or a sequence of response strings for batch processing.
                query: Either a single query string or a sequence of query strings for batch processing.
                context: Either a single context string or a sequence of context strings for batch processing.
                prompt: Either a single prompt string or a sequence of prompt strings for batch processing.
                       This parameter is mainly used for replacing the default prompt format entirely.
                form_prompt: Optional function to format the prompt. Cannot be provided together with prompt.
                       The function should take query and context as parameters and return a formatted prompt string.
                       Note: If you need to include a system prompt or any other special instructions, you must
                       incorporate them directly in your custom form_prompt function implementation.
                       
           Returns:
                TrustworthyRAGScore | list[TrustworthyRAGScore]: A single score or a list of scores
                if batch processing was used. Each score contains evaluation metrics for the response.
        """
        # Validate that prompt and form_prompt are not provided at the same time
        if prompt is not None and form_prompt is not None:
            raise ValidationError("'prompt' and 'form_prompt' cannot be provided at the same time. Use either one, not both.")

        # Use the class method as default if form_prompt is not provided and prompt is None
        if prompt is None and form_prompt is None:
            form_prompt = self._default_prompt_formatter

        # Check if we're dealing with batch inputs
        is_batch = False
        batch_size = None

        # Check if any of the inputs are sequences (for batch processing)
        if isinstance(response, Sequence) and not isinstance(response, str):
            is_batch = True
            batch_size = len(response)
        elif isinstance(query, Sequence) and not isinstance(query, str):
            is_batch = True
            batch_size = len(query)
        elif isinstance(context, Sequence) and not isinstance(context, str):
            is_batch = True
            batch_size = len(context)
        elif isinstance(prompt, Sequence) and not isinstance(prompt, str):
            is_batch = True
            batch_size = len(prompt)

        # Handle single input case
        if not is_batch:
            # Validate that either prompt or both query and context are provided
            if prompt is None and (query is None or context is None):
                raise ValidationError("Either 'prompt' or both 'query' and 'context' must be provided")

            if prompt is None:
                prompt = form_prompt(query, context)

            # Validate parameter types
            for param_name, param_value in [
                ("prompt", prompt),
                ("query", query),
                ("context", context),
                ("response", response),
            ]:
                if not isinstance(param_value, str):
                    raise ValidationError(f"'{param_name}' must be a string")

            # Validate inputs for evaluations
            is_valid, error_message = self._validate_inputs(
                query=query,
                context=context,
                response=response,
            )
            if not is_valid:
                raise ValidationError(error_message)

            score_result = self._event_loop.run_until_complete(
                self._score_async(
                    response=response,
                    prompt=prompt,
                    query=query,
                    context=context,
                )
            )

            # Return the score with the updated TrustworthyRAGScore format
            return TrustworthyRAGScore(**score_result)

        # Handle batch input case
        else:
            # Ensure all batch inputs have the same length
            batch_inputs = []
            for param_name, param_value in [
                ("response", response),
                ("query", query),
                ("context", context),
                ("prompt", prompt),
            ]:
                if param_value is not None:
                    if isinstance(param_value, Sequence) and not isinstance(param_value, str):
                        if batch_size is not None and len(param_value) != batch_size:
                            raise ValidationError(f"All batch inputs must have the same length, but '{param_name}' has length {len(param_value)} while expected {batch_size}")
                        batch_inputs.append((param_name, param_value))
                    elif isinstance(param_value, str):
                        # Convert single string to a list of the same string repeated batch_size times
                        batch_inputs.append((param_name, [param_value] * batch_size))
                    else:
                        raise ValidationError(f"'{param_name}' must be a string or a sequence of strings")

            # Create dictionaries for each batch item
            batch_items = [{} for _ in range(batch_size)]
            for param_name, param_values in batch_inputs:
                for i, value in enumerate(param_values):
                    batch_items[i][param_name] = value

            # Process each batch item
            processed_responses = []
            processed_prompts = []
            processed_queries = []
            processed_contexts = []

            for item in batch_items:
                item_response = item.get("response")
                item_prompt = item.get("prompt")
                item_query = item.get("query")
                item_context = item.get("context")

                # Validate that either prompt or both query and context are provided
                if item_prompt is None and (item_query is None or item_context is None):
                    raise ValidationError("Either 'prompt' or both 'query' and 'context' must be provided for each batch item")

                # Generate prompt if not provided
                if item_prompt is None:
                    item_prompt = form_prompt(item_query, item_context)

                # Validate inputs for evaluations
                is_valid, error_message = self._validate_inputs(
                    query=item_query,
                    context=item_context,
                    response=item_response,
                )
                if not is_valid:
                    raise ValidationError(error_message)

                processed_responses.append(item_response)
                processed_prompts.append(item_prompt)
                processed_queries.append(item_query)
                processed_contexts.append(item_context)

            # Run batch processing
            scores = self._event_loop.run_until_complete(
                self._batch_score(
                    responses=processed_responses,
                    prompts=processed_prompts,
                    queries=processed_queries,
                    contexts=processed_contexts,
                    capture_exceptions=False,
                )
            )

            # Convert scores to TrustworthyRAGScore format
            return [TrustworthyRAGScore(**score) for score in scores]

    def get_evals(self) -> list[Eval]:
        """
        Returns a copy of the evaluation criteria currently being used by this TrustworthyRAG instance.
        
        Returns:
            A list of Eval objects with the current evaluation criteria
        """
        return self._evals.copy()

    def _validate_inputs(
        self,
        query: Optional[Union[str, Sequence[str]]] = None,
        context: Optional[Union[str, Sequence[str]]] = None,
        response: Optional[Union[str, Sequence[str]]] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Validate that all required inputs for evaluations are present.
        
        Args:
            query: The user query or sequence of queries
            context: The context used for RAG or sequence of contexts
            response: The response to evaluate or sequence of responses
            
        Returns:
            A tuple of (is_valid, error_message)
            - is_valid: True if all required inputs are present, False otherwise
            - error_message: Error message if validation fails, None otherwise
        """
        # For validation purposes, we don't need to check for sequences again
        # as the calling methods will have already raised errors for sequences

        # Determine if we're validating for generate (no response) or score (with response)
        is_generate = response is None

        for eval_obj in self._evals:
            # Check query identifier
            if eval_obj.query_identifier and (query is None):
                return False, f"Missing required input 'query' for evaluation '{eval_obj.name}'"

            # Check context identifier
            if eval_obj.context_identifier and (context is None):
                return False, f"Missing required input 'context' for evaluation '{eval_obj.name}'"

            # Check response identifier (skip in generate mode)
            if not is_generate and eval_obj.response_identifier and (response is None):
                return False, f"Missing required input 'response' for evaluation '{eval_obj.name}'"

        return True, None

    def _default_prompt_formatter(
        self,
        query: Union[str, Sequence[str]],
        context: Union[str, Sequence[str]]
    ) -> str:
        """
        Format a standard RAG prompt using the provided components.
        
        Note: This default formatter does not include a system prompt. If you need to include
        a system prompt or other special instructions, use a custom form_prompt function.
        
        Args:
            query: The user's question or request, or a sequence of queries
            context: Retrieved context/documents to help answer the query, or a sequence of contexts
            
        Returns:
            A formatted prompt string ready to be sent to the model
        """
        # Validate parameter types
        for param_name, param_value in [
            ("query", query),
            ("context", context),
        ]:
            if param_value is not None:
                # Validate parameter type
                if not isinstance(param_value, str):
                    raise ValidationError(f"'{param_name}' must be a string")

        # Start with prompt parts
        prompt_parts = []

        # Add context
        prompt_parts.append("Context information is below.\n")
        prompt_parts.append("---------------------\n")
        prompt_parts.append(f"{context.strip()}\n")
        prompt_parts.append("---------------------\n")

        # Add user query
        prompt_parts.append(f"User: {query.strip()}\n")

        # Add assistant response starter
        prompt_parts.append("Assistant: ")

        return "\n".join(prompt_parts)

    @handle_tlm_exceptions("TrustworthyRAGResponse")
    async def _generate_async(
        self,
        *,
        prompt: str,  # Already processed in the generate method
        query: Optional[str] = None,
        context: Optional[str] = None,
        timeout: Optional[float] = None,
        capture_exceptions: bool = False,  # Added for exception handling
        batch_index: Optional[int] = None,  # Added for exception handling
    ) -> TrustworthyRAGResponse:
        """
        Private asynchronous method to get response and trustworthiness score from TLM.

        Args:
            prompt (str): prompt for the TLM. If a sequence was provided, it has been processed
                         in the generate method to extract the first element.
            query (str, optional): The query string.
            context (str, optional): The context string.
            timeout (float, optional): timeout (in seconds) to run the prompt, defaults to None (no timeout)
        Returns:
            TrustworthyRAGResponse: [TrustworthyRAGResponse](#class-tlmresponse) object containing the response and trustworthiness score.
        """
        response_json = await asyncio.wait_for(
            tlm_rag_generate(
                api_key=self._api_key,
                prompt=prompt,
                quality_preset=self._quality_preset,
                options=self._options,
                rate_handler=self._rate_handler,
                query=query,
                context=context,
                evals=self._evals,
            ),
            timeout=timeout,
        )

        return response_json

    @handle_tlm_exceptions("TrustworthyRAGScore")
    async def _score_async(
        self,
        *,
        response: str,
        prompt: str,  # Already processed in the score method
        query: Optional[str] = None,
        context: Optional[str] = None,
        timeout: Optional[float] = None,
        capture_exceptions: bool = False,  # Added for exception handling
        batch_index: Optional[int] = None,  # Added for exception handling
    ) -> TrustworthyRAGScore:
        """
        Private asynchronous method to get trustworthiness score from TLM.

        Args:
            response (str): response to score
            prompt (str): prompt for the TLM. If a sequence was provided, it has been processed
                         in the score method to extract the first element.
            query (str, optional): The query string.
            context (str, optional): The context string.
            timeout (float, optional): timeout (in seconds) to run the prompt, defaults to None (no timeout)
        Returns:
            TrustworthyRAGScore: [TrustworthyRAGScore](#class-tlmresponse) object containing the trustworthiness score.
        """
        response_json = await asyncio.wait_for(
            tlm_rag_score(
                api_key=self._api_key,
                response=response,
                prompt=prompt,
                quality_preset=self._quality_preset,
                options=self._options,
                rate_handler=self._rate_handler,
                query=query,
                context=context,
                evals=self._evals,
            ),
            timeout=timeout,
        )

        return response_json

    async def _batch_generate(
        self,
        prompts: Sequence[str],
        queries: Sequence[Optional[str]],
        contexts: Sequence[Optional[str]],
        capture_exceptions: bool = False,
    ) -> list[TrustworthyRAGResponse]:
        """Run a batch of prompts through TrustworthyRAG and get responses/scores for each prompt in the batch.
        The list returned will have the same length as the input list.

        Args:
            prompts (Sequence[str]): list of prompts to run
            queries (Sequence[Optional[str]]): list of queries corresponding to each prompt
            contexts (Sequence[Optional[str]]): list of contexts corresponding to each prompt
            capture_exceptions (bool): if ``True``, the returned list will contain TrustworthyRAGResponse objects 
                with error messages in place of the response for any errors or timeout when processing a particular prompt from the batch.
                If ``False``, this entire method will raise an exception if TrustworthyRAG fails to produce a result for any prompt in the batch.

        Returns:
            list[TrustworthyRAGResponse]: TrustworthyRAG responses/scores for each prompt (in supplied order)
        """
        if capture_exceptions:
            per_query_timeout, per_batch_timeout = self._timeout, None
        else:
            per_query_timeout, per_batch_timeout = None, self._timeout

        # run batch of TrustworthyRAG
        rag_responses = await self._batch_async(
            [
                self._generate_async(
                    prompt=prompt,
                    query=query,
                    context=context,
                    timeout=per_query_timeout,
                    capture_exceptions=capture_exceptions,
                    batch_index=batch_index,
                )
                for batch_index, (prompt, query, context) in enumerate(zip(prompts, queries, contexts))
            ],
            per_batch_timeout,
        )

        return cast(list[TrustworthyRAGResponse], rag_responses)

    async def _batch_score(
        self,
        responses: Sequence[str],
        prompts: Sequence[str],
        queries: Sequence[Optional[str]],
        contexts: Sequence[Optional[str]],
        capture_exceptions: bool = False,
    ) -> list[TrustworthyRAGScore]:
        """Run batch of TrustworthyRAG score evaluations.

        capture_exceptions behavior:
        - If true, the list will contain None in place of the response for any errors or timeout processing some inputs.
        - Otherwise, the method will raise an exception for any errors or timeout processing some inputs.

        capture_exceptions interaction with timeout:
        - If true, timeouts are applied on a per-query basis (i.e. some queries may succeed while others fail)
        - If false, a single timeout is applied to the entire batch (i.e. all queries will fail if the timeout is reached)

        Args:
            responses (Sequence[str]): list of responses to evaluate
            prompts (Sequence[str]): list of prompts corresponding to each response
            queries (Sequence[Optional[str]]): list of queries corresponding to each prompt
            contexts (Sequence[Optional[str]]): list of contexts corresponding to each prompt
            capture_exceptions (bool): if True, the returned list will contain TrustworthyRAGScore objects 
                with error messages in place of the score for any errors or timeout when processing a particular item from the batch.

        Returns:
            list[TrustworthyRAGScore]: TrustworthyRAG scores for each response (in supplied order).
        """
        if capture_exceptions:
            per_query_timeout, per_batch_timeout = self._timeout, None
        else:
            per_query_timeout, per_batch_timeout = None, self._timeout

        # run batch of TrustworthyRAG score evaluations
        rag_scores = await self._batch_async(
            [
                self._score_async(
                    response=response,
                    prompt=prompt,
                    query=query,
                    context=context,
                    timeout=per_query_timeout,
                    capture_exceptions=capture_exceptions,
                    batch_index=batch_index,
                )
                for batch_index, (response, prompt, query, context) in enumerate(zip(responses, prompts, queries, contexts))
            ],
            per_batch_timeout,
        )

        return cast(list[TrustworthyRAGScore], rag_scores)

    async def _batch_async(
        self,
        rag_coroutines: Sequence[Coroutine[None, None, Union[TrustworthyRAGResponse, TrustworthyRAGScore]]],
        batch_timeout: Optional[float] = None,
    ) -> Sequence[Union[TrustworthyRAGResponse, TrustworthyRAGScore]]:
        """Runs batch of TrustworthyRAG queries.

        Args:
            rag_coroutines (Sequence[Coroutine]): list of query coroutines to run, returning TrustworthyRAGResponse or TrustworthyRAGScore
            batch_timeout (Optional[float], optional): timeout (in seconds) to run all queries, defaults to None (no timeout)

        Returns:
            Sequence[Union[TrustworthyRAGResponse, TrustworthyRAGScore]]: list of coroutine results, with preserved order
        """
        rag_query_tasks = [asyncio.create_task(rag_coro) for rag_coro in rag_coroutines]

        if self._verbose:
            gather_task = tqdm_asyncio.gather(
                *rag_query_tasks,
                total=len(rag_query_tasks),
                desc="Querying TrustworthyRAG...",
                bar_format="{desc} {percentage:3.0f}%|{bar}|",
            )
        else:
            gather_task = asyncio.gather(*rag_query_tasks)  # type: ignore[assignment]

        wait_task = asyncio.wait_for(gather_task, timeout=batch_timeout)
        try:
            return cast(
                Sequence[Union[TrustworthyRAGResponse, TrustworthyRAGScore]],
                await wait_task,
            )
        except Exception:
            # if exception occurs while awaiting batch results, cancel remaining tasks
            for query_task in rag_query_tasks:
                query_task.cancel()

            # await remaining tasks to ensure they are cancelled
            await asyncio.gather(*rag_query_tasks, return_exceptions=True)

            raise

    def try_generate(
        self,
        *,
        query: Sequence[str],
        context: Sequence[str],
        prompt: Optional[Sequence[str]] = None,
        form_prompt: Optional[Callable[[str, str], str]] = None
    ) -> list[TrustworthyRAGResponse]:
        """
        Gets response and evaluation scores for a batch of queries and contexts, handling any failures (errors or timeouts).

        The list returned will have the same length as the input lists. If there are any
        failures (errors or timeouts) processing some inputs, the TrustworthyRAGResponse objects in the returned list 
        will contain error messages and retryability information instead of the usual response.

        This is the recommended way to run TrustworthyRAG over large datasets with many queries.
        It ensures partial results are preserved, even if some individual TrustworthyRAG calls over the dataset fail.

        Args:
            query (Sequence[str]): list of queries for the TrustworthyRAG
            context (Sequence[str]): list of contexts for the TrustworthyRAG
            prompt (Sequence[str], optional): list of prompts for the TrustworthyRAG
            form_prompt (Callable[[str, str], str], optional): function to format the prompt
        Returns:
            list[TrustworthyRAGResponse]: list of TrustworthyRAGResponse objects containing the response and evaluation scores.
                The returned list will always have the same length as the input lists.
                In case of TrustworthyRAG failure on any input (due to timeouts or other errors),
                the return list will include a TrustworthyRAGResponse with an error message and retryability information 
                instead of the usual TrustworthyRAGResponse for that failed input.
        """
        # Validate that prompt and form_prompt are not provided at the same time
        if prompt is not None and form_prompt is not None:
            raise ValidationError("'prompt' and 'form_prompt' cannot be provided at the same time. Use either one, not both.")

        # Use the class method as default if form_prompt is not provided and prompt is None
        if prompt is None and form_prompt is None:
            form_prompt = self._default_prompt_formatter

        # Ensure all batch inputs have the same length
        batch_size = len(query)
        if len(context) != batch_size:
            raise ValidationError(f"All batch inputs must have the same length, but 'context' has length {len(context)} while expected {batch_size}")
        if prompt is not None and len(prompt) != batch_size:
            raise ValidationError(f"All batch inputs must have the same length, but 'prompt' has length {len(prompt)} while expected {batch_size}")

        # Process each batch item
        processed_prompts = []
        processed_queries = []
        processed_contexts = []

        for i in range(batch_size):
            item_query = query[i]
            item_context = context[i]
            item_prompt = prompt[i] if prompt is not None else None

            # Validate that either prompt or both query and context are provided
            if item_prompt is None and (item_query is None or item_context is None):
                raise ValidationError("Either 'prompt' or both 'query' and 'context' must be provided for each batch item")

            # Generate prompt if not provided
            if item_prompt is None:
                item_prompt = form_prompt(item_query, item_context)

            # Validate inputs for evaluations
            is_valid, error_message = self._validate_inputs(
                query=item_query,
                context=item_context,
            )
            if not is_valid:
                raise ValidationError(error_message)

            processed_prompts.append(item_prompt)
            processed_queries.append(item_query)
            processed_contexts.append(item_context)

        # Run batch processing with exception handling
        responses = self._event_loop.run_until_complete(
            self._batch_generate(
                prompts=processed_prompts,
                queries=processed_queries,
                contexts=processed_contexts,
                capture_exceptions=True,
            )
        )

        # Convert responses to TrustworthyRAGResponse format
        return [TrustworthyRAGResponse(**response) for response in responses]

    def try_score(
        self,
        *,
        response: Sequence[str],
        query: Sequence[str],
        context: Sequence[str],
        prompt: Optional[Sequence[str]] = None,
        form_prompt: Optional[Callable[[str, str], str]] = None
    ) -> list[TrustworthyRAGScore]:
        """
        Scores a batch of responses against queries and contexts, handling any failures (errors or timeouts).

        The list returned will have the same length as the input lists. If there are any
        failures (errors or timeouts) processing some inputs, the TrustworthyRAGScore objects in the returned list 
        will contain error messages and retryability information instead of the usual score.

        This is the recommended way to run TrustworthyRAG scoring over large datasets.
        It ensures partial results are preserved, even if some individual TrustworthyRAG calls over the dataset fail.

        Args:
            response (Sequence[str]): list of responses to score
            query (Sequence[str]): list of queries for the TrustworthyRAG
            context (Sequence[str]): list of contexts for the TrustworthyRAG
            prompt (Sequence[str], optional): list of prompts for the TrustworthyRAG
            form_prompt (Callable[[str, str], str], optional): function to format the prompt
        Returns:
            list[TrustworthyRAGScore]: list of TrustworthyRAGScore objects containing the evaluation scores.
                The returned list will always have the same length as the input lists.
                In case of TrustworthyRAG failure on any input (due to timeouts or other errors),
                the return list will include a TrustworthyRAGScore with an error message and retryability information 
                instead of the usual TrustworthyRAGScore for that failed input.
        """
        # Validate that prompt and form_prompt are not provided at the same time
        if prompt is not None and form_prompt is not None:
            raise ValidationError("'prompt' and 'form_prompt' cannot be provided at the same time. Use either one, not both.")

        # Use the class method as default if form_prompt is not provided and prompt is None
        if prompt is None and form_prompt is None:
            form_prompt = self._default_prompt_formatter

        # Ensure all batch inputs have the same length
        batch_size = len(response)
        if len(query) != batch_size:
            raise ValidationError(f"All batch inputs must have the same length, but 'query' has length {len(query)} while expected {batch_size}")
        if len(context) != batch_size:
            raise ValidationError(f"All batch inputs must have the same length, but 'context' has length {len(context)} while expected {batch_size}")
        if prompt is not None and len(prompt) != batch_size:
            raise ValidationError(f"All batch inputs must have the same length, but 'prompt' has length {len(prompt)} while expected {batch_size}")

        # Process each batch item
        processed_responses = []
        processed_prompts = []
        processed_queries = []
        processed_contexts = []

        for i in range(batch_size):
            item_response = response[i]
            item_query = query[i]
            item_context = context[i]
            item_prompt = prompt[i] if prompt is not None else None

            # Validate that either prompt or both query and context are provided
            if item_prompt is None and (item_query is None or item_context is None):
                raise ValidationError("Either 'prompt' or both 'query' and 'context' must be provided for each batch item")

            # Generate prompt if not provided
            if item_prompt is None:
                item_prompt = form_prompt(item_query, item_context)

            # Validate inputs for evaluations
            is_valid, error_message = self._validate_inputs(
                query=item_query,
                context=item_context,
                response=item_response,
            )
            if not is_valid:
                raise ValidationError(error_message)

            processed_responses.append(item_response)
            processed_prompts.append(item_prompt)
            processed_queries.append(item_query)
            processed_contexts.append(item_context)

        # Run batch processing with exception handling
        scores = self._event_loop.run_until_complete(
            self._batch_score(
                responses=processed_responses,
                prompts=processed_prompts,
                queries=processed_queries,
                contexts=processed_contexts,
                capture_exceptions=True,
            )
        )

        # Convert scores to TrustworthyRAGScore format
        return [TrustworthyRAGScore(**score) for score in scores]


class Eval:
    """
    Class representing an evaluation for TrustworthyRAG.
    
    Args:
        name: The name of the evaluation
        criteria: The evaluation criteria text
        query_identifier: Identifier for the query input (default: None)
        context_identifier: Identifier for the context input (default: None)
        response_identifier: Identifier for the response input (default: None)
    """
    def __init__(
        self,
        name: str,
        criteria: str,
        query_identifier=None, # "Query"
        context_identifier=None, # "Document"
        response_identifier=None, # "Response"
    ):
        self.name = name
        self.criteria = criteria
        self.query_identifier = query_identifier
        self.context_identifier = context_identifier
        self.response_identifier = response_identifier


DEFAULT_EVALS = [
    {
        "name": "context_informativeness",
        "criteria": "Determine whether the Context contains all of the information necessary to answer the Query. A good Context will contain every piece of information that would be required to answer the Query without prior knowledge. A bad Context would have none of this information, perhaps being unrelated to the Query.",
        "query_identifier": "Query",
        "context_identifier": "Context",
        "response_identifier": None
    },
    {
        "name": "context_clarity",
        "criteria": "Determine whether the Context is clear and consistent. Check if the information is confusing, overly complex, or contains self-contradictions. A good Context should be easy to understand and internally consistent. A bad Context would be difficult to follow, unnecessarily complex, or have contradicting statements.",
        "query_identifier": None,
        "context_identifier": "Context",
        "response_identifier": None
    },
    {
        "name": "response_helpfulness",
        "criteria": "Determine whether the Response is helpful in addressing the Query. Check if the Response directly answers the Query, provides useful information, or appropriately indicates when it cannot provide an answer. A good Response will either give a clear, relevant answer or explicitly acknowledge its limitations (e.g., 'I don't know' or 'I need more information'). A bad Response would be irrelevant, evasive, or provide misleading information when it should acknowledge uncertainty.",
        "query_identifier": "Query",
        "context_identifier": None,
        "response_identifier": "Response"
    },
    {
        "name": "response_grounding",
        "criteria": "Determine whether the Response is fully grounded in the Context. Check if all claims and information in the Response can be directly supported by information provided in the Context. A good Response will only contain information that is present in or can be reasonably inferred from the Context. A bad Response would include claims or information not supported by the Context.",
        "query_identifier": None,
        "context_identifier": "Context",
        "response_identifier": "Response"
    },
    {
        "name": "query_clarity",
        "criteria": "Determine whether the Query is clear and complete. Check if the Query is overly complex, vague, or missing key contextual information (such as software versions, specific products, or other relevant details needed to provide an accurate response). A good Query should be specific, focused, and include all necessary context. A bad Query would be ambiguous, overly broad, or lack critical details needed to properly address the user's needs.",
        "query_identifier": "Query",
        "context_identifier": None,
        "response_identifier": None
    },
    {
        "name": "response_sentiment",
        "criteria": "Determine whether the Response is written in a positive tone. Answer Yes or No only: Yes if the tone is positive, No if it is negative or neutral.",
        "query_identifier": None,
        "context_identifier": None,
        "response_identifier": "Response"
    }
]


def get_default_evals() -> list[Eval]:
    """
    Get default evaluation criteria from the client-side defaults.
    
    Args:
        api_key: Not used, kept for backward compatibility
        
    Returns:
        A list of Eval objects with default evaluation criteria
    """
    return [
        Eval(
            name=eval_config["name"],
            criteria=eval_config["criteria"],
            query_identifier=eval_config.get("query_identifier"),
            context_identifier=eval_config.get("context_identifier"),
            response_identifier=eval_config.get("response_identifier"),
        )
        for eval_config in DEFAULT_EVALS
    ]

# Define the response types first
class EvaluationMetric(TypedDict):
    """Evaluation metric with score and optional logs.

    Attributes:
        score (float, optional): score between 0-1 corresponding to the evaluation metric.
        A higher score indicates a higher rating for the specific evaluation criteria being measured.

        log (dict, optional): additional logs and metadata returned from the LLM call, only if the `log` key was specified in TLMOptions.
    """

    score: Optional[float]
    log: NotRequired[dict[str, Any]]


class TrustworthyRAGResponse(TypedDict, total=False):
    """Response from TrustworthyRAG with generated text and evaluation scores.

    Attributes:
        response (str, optional): The generated response text.
        
        Additional keys: Each evaluation metric appears as a top-level key in the dictionary,
        with values following the EvaluationMetric structure (containing score and optional log).
        
    Example:
        ```python
        {
            "response": "<response text>",
            "trustworthiness": {
                "score": 0.92,
                "log": {"explanation": "Did not find a reason to doubt trustworthiness."}
            },
            "context_informativeness": {
                "score": 0.65
            },
            ...
        }
        ```
    """

    response: Optional[str]


class TrustworthyRAGScore(TypedDict, total=False):
    """Evaluation scores for an existing RAG response.

    Attributes:
        Each evaluation metric appears as a top-level key in the dictionary,
        with values following the EvaluationMetric structure (containing score and optional log).
        
    Example:
        ```python
        {
            "trustworthiness": {
                "score": 0.92,
                "log": {"explanation": "Did not find a reason to doubt trustworthiness."}
            },
            "context_informativeness": {
                "score": 0.65
            },
            ...
        }
        ```
    """


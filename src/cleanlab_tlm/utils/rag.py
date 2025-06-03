"""
Real-time Evals for Retrieval-Augmented Generation (RAG) systems, powered by Cleanlab's Trustworthy Language Model (TLM).

This module combines Cleanlab's trustworthiness scores for each RAG response with additional Evals for other RAG components (such as the retrieved context).

You can also customize Evals for your use-case. Each Eval provides real-time detection of quality issues in your RAG application based on the: user query, retrieved context (documents), and/or LLM-generated response.

For RAG use-cases, we recommend using this module's `TrustworthyRAG` object in place of the basic `TLM` object.

This feature is in Beta, contact us if you encounter issues.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import (
    # lazydocs: ignore
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Union,
    cast,
)

from tqdm.asyncio import tqdm_asyncio
from typing_extensions import NotRequired, TypedDict

from cleanlab_tlm.errors import ValidationError
from cleanlab_tlm.internal.api import api
from cleanlab_tlm.internal.base import BaseTLM
from cleanlab_tlm.internal.constants import (
    _DEFAULT_TLM_QUALITY_PRESET,
    _TLM_EVAL_CONTEXT_IDENTIFIER_KEY,
    _TLM_EVAL_CRITERIA_KEY,
    _TLM_EVAL_NAME_KEY,
    _TLM_EVAL_QUERY_IDENTIFIER_KEY,
    _TLM_EVAL_RESPONSE_IDENTIFIER_KEY,
    _TLM_MAX_RETRIES,
    _VALID_TLM_QUALITY_PRESETS_RAG,
)
from cleanlab_tlm.internal.exception_handling import handle_tlm_exceptions
from cleanlab_tlm.internal.validation import (
    tlm_score_process_response_and_kwargs,
    validate_rag_inputs,
)

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from cleanlab_tlm.internal.types import TLMQualityPreset
    from cleanlab_tlm.tlm import TLMOptions


class TrustworthyRAG(BaseTLM):
    """
    Real-time Evals for Retrieval-Augmented Generation (RAG) systems, powered by Cleanlab's Trustworthy Language Model (TLM).

    For RAG use-cases, we recommend using this object in place of the basic `TLM` object. You can use `TrustworthyRAG` to either `score` an existing RAG response (from any LLM) based on user query and retrieved context, or to both `generate` the RAG response and score it simultaneously.

    This object combines Cleanlab's trustworthiness scores for each RAG response with additional Evals for other RAG components (such as the retrieved context).

    You can also customize Evals for your use-case. Each Eval provides real-time detection of quality issues in your RAG application based on the: user query, retrieved context (documents), and/or LLM-generated response.

    Most arguments for this `TrustworthyRAG()` class are similar to those for [TLM](../tlm/#class-tlm), the
    differences are described below. For details about each argument, refer to the [TLM](../tlm/#class-tlm) documentation.

    Args:
        quality_preset ({"base", "low", "medium"}, default = "medium"): an optional preset configuration to control
            the quality of generated LLM responses and trustworthiness scores vs. latency/costs.

        api_key (str, optional): API key for accessing TLM. If not provided, this client will
            attempt to use the CLEANLAB_TLM_API_KEY environment variable.

        options ([TLMOptions](../tlm/#class-tlmoptions), optional): a typed dict of advanced configurations you can optionally specify.
            The "custom_eval_criteria" key for [TLM](../tlm/#class-tlm) is not supported for `TrustworthyRAG`, you can instead specify `evals`.

        timeout (float, optional): timeout (in seconds) to apply to each request.

        verbose (bool, optional): whether to print outputs during execution, i.e. show a progress bar when processing a batch of data.

        evals (list[Eval], optional): additional evaluation criteria to check for, in addition to response trustworthiness.
            If not specified, default evaluations will be used (access these via [get_default_evals](#function-get_default_evals)).
            To come up with your custom `evals`, we recommend you first run [get_default_evals()](#function-get_default_evals) and then add/remove/modify the returned list.
            Each [Eval](#class-eval) in this list provides real-time detection of specific issues in your RAG application based on the user query, retrieved context (documents), and/or LLM-generated response.
            Set this to an empty list to only score response trustworthiness without additional evaluations.
    """

    def __init__(
        self,
        quality_preset: TLMQualityPreset = _DEFAULT_TLM_QUALITY_PRESET,
        *,
        api_key: Optional[str] = None,
        options: Optional[TLMOptions] = None,
        timeout: Optional[float] = None,
        verbose: Optional[bool] = None,
        evals: Optional[list[Eval]] = None,
    ) -> None:
        """
        lazydocs: ignore
        """
        # Initialize base class
        super().__init__(
            quality_preset=quality_preset,
            valid_quality_presets=_VALID_TLM_QUALITY_PRESETS_RAG,
            support_custom_eval_criteria=False,
            api_key=api_key,
            options=options,
            timeout=timeout,
            verbose=verbose,
        )

        # TrustworthyRAG-specific initialization
        # If evals not provided, use the default evals defined in this file
        if evals is None:
            self._evals = [
                Eval(
                    name=cast(str, eval_config[_TLM_EVAL_NAME_KEY]),
                    criteria=cast(str, eval_config[_TLM_EVAL_CRITERIA_KEY]),
                    query_identifier=eval_config.get(_TLM_EVAL_QUERY_IDENTIFIER_KEY),
                    context_identifier=eval_config.get(_TLM_EVAL_CONTEXT_IDENTIFIER_KEY),
                    response_identifier=eval_config.get(_TLM_EVAL_RESPONSE_IDENTIFIER_KEY),
                )
                for eval_config in _DEFAULT_EVALS
            ]
        else:
            # validate that evals is a list of Eval objects
            if not isinstance(evals, list) or any(not isinstance(ev, Eval) for ev in evals):
                raise ValidationError("'evals' must be a list of Eval objects")

            self._evals = evals

    def score(
        self,
        *,
        response: Union[str, Sequence[str]],
        query: Union[str, Sequence[str]],
        context: Union[str, Sequence[str]],
        prompt: Optional[Union[str, Sequence[str]]] = None,
        form_prompt: Optional[Callable[[str, str], str]] = None,
    ) -> Union[TrustworthyRAGScore, list[TrustworthyRAGScore]]:
        """
        Evaluate an existing RAG system's response to a given user query and retrieved context.

        Args:
             response (str | Sequence[str]): A response (or list of multiple responses) from your LLM/RAG system.
             query (str | Sequence[str]): The user query (or list of multiple queries) that was used to generate the response.
             context (str | Sequence[str]): The context (or list of multiple contexts) that was retrieved from the RAG Knowledge Base and used to generate the response.
             prompt (str | Sequence[str], optional): Optional prompt (or list of multiple prompts) representing the actual inputs (combining query, context, and system instructions into one string) to the LLM that generated the response.
             form_prompt (Callable[[str, str], str], optional): Optional function to format the prompt based on query and context. Cannot be provided together with prompt, provide one or the other.
                    This function should take query and context as parameters and return a formatted prompt string.
                    If not provided, a default prompt formatter will be used.
                    To include a system prompt or any other special instructions for your LLM,
                    incorporate them directly in your custom `form_prompt()` function definition.

        Returns:
             TrustworthyRAGScore | list[TrustworthyRAGScore]: [TrustworthyRAGScore](#class-trustworthyragscore) object containing evaluation metrics.
                 If multiple inputs were provided in lists, a list of TrustworthyRAGScore objects is returned, one for each set of inputs.
        """
        if prompt is None and form_prompt is None:
            form_prompt = TrustworthyRAG._default_prompt_formatter

        formatted_prompts = validate_rag_inputs(
            query=query,
            context=context,
            response=response,
            prompt=prompt,
            form_prompt=form_prompt,
            evals=self._evals,
            is_generate=False,
        )

        # Support constrain_outputs later
        processed_responses = tlm_score_process_response_and_kwargs(formatted_prompts, response, None, {})

        # Check if we're handling a batch or a single item
        if isinstance(query, str) and isinstance(context, str) and isinstance(processed_responses, dict):
            return self._event_loop.run_until_complete(
                self._score_async(
                    response=processed_responses,
                    prompt=formatted_prompts,
                    query=query,
                    context=context,
                    timeout=self._timeout,
                )
            )

        # Batch processing
        return self._event_loop.run_until_complete(
            self._batch_score(
                responses=cast(Sequence[dict[str, Any]], processed_responses),
                prompts=formatted_prompts,
                queries=query,
                contexts=context,
            )
        )

    async def score_async(
        self,
        *,
        response: Union[str, Sequence[str]],
        query: Union[str, Sequence[str]],
        context: Union[str, Sequence[str]],
        prompt: Optional[Union[str, Sequence[str]]] = None,
        form_prompt: Optional[Callable[[str, str], str]] = None,
    ) -> Union[TrustworthyRAGScore, list[TrustworthyRAGScore]]:
        """
        Asynchronously evaluate an existing RAG system's response to a given user query and retrieved context.

        Args:
             response (str | Sequence[str]): A response (or list of multiple responses) from your LLM/RAG system.
             query (str | Sequence[str]): The user query (or list of multiple queries) that was used to generate the response.
             context (str | Sequence[str]): The context (or list of multiple contexts) that was retrieved from the RAG Knowledge Base and used to generate the response.
             prompt (str | Sequence[str], optional): Optional prompt (or list of multiple prompts) representing the actual inputs (combining query, context, and system instructions into one string) to the LLM that generated the response.
             form_prompt (Callable[[str, str], str], optional): Optional function to format the prompt based on query and context. Cannot be provided together with prompt, provide one or the other.
                    This function should take query and context as parameters and return a formatted prompt string.
                    If not provided, a default prompt formatter will be used.
                    To include a system prompt or any other special instructions for your LLM,
                    incorporate them directly in your custom `form_prompt()` function definition.

        Returns:
             TrustworthyRAGScore | list[TrustworthyRAGScore]: [TrustworthyRAGScore](#class-trustworthyragscore) object containing evaluation metrics.
                 If multiple inputs were provided in lists, a list of TrustworthyRAGScore objects is returned, one for each set of inputs.
        """
        if prompt is None and form_prompt is None:
            form_prompt = TrustworthyRAG._default_prompt_formatter

        formatted_prompts = validate_rag_inputs(
            query=query,
            context=context,
            response=response,
            prompt=prompt,
            form_prompt=form_prompt,
            evals=self._evals,
            is_generate=False,
        )

        # Support constrain_outputs later
        processed_responses = tlm_score_process_response_and_kwargs(formatted_prompts, response, None, {})

        # Check if we're handling a batch or a single item
        if isinstance(query, str) and isinstance(context, str) and isinstance(processed_responses, dict):
            return await self._score_async(
                response=processed_responses,
                prompt=formatted_prompts,
                query=query,
                context=context,
                timeout=self._timeout,
            )

        # Batch processing
        return await self._batch_score(
            responses=cast(Sequence[dict[str, Any]], processed_responses),
            prompts=formatted_prompts,
            queries=query,
            contexts=context,
        )

    def generate(
        self,
        *,
        query: Union[str, Sequence[str]],
        context: Union[str, Sequence[str]],
        prompt: Optional[Union[str, Sequence[str]]] = None,
        form_prompt: Optional[Callable[[str, str], str]] = None,
    ) -> Union[TrustworthyRAGResponse, list[TrustworthyRAGResponse]]:
        """
        Generate a RAG response and evaluate/score it simultaneously.

        You can use this method in place of the generator LLM in your RAG application (no change to your prompts needed).
        It will both produce the response based on query/context and the corresponding evaluations computed by [score()](#method-score).

        This method relies on the same arguments as [score()](#method-score), except you should not provide a `response`.

        Returns:
             TrustworthyRAGResponse | list[TrustworthyRAGResponse]: [TrustworthyRAGResponse](#class-trustworthyragresponse) object containing the generated response text and corresponding evaluation scores.
        """
        if prompt is None and form_prompt is None:
            form_prompt = TrustworthyRAG._default_prompt_formatter

        formatted_prompts = validate_rag_inputs(
            query=query,
            context=context,
            prompt=prompt,
            form_prompt=form_prompt,
            evals=self._evals,
            is_generate=True,
        )

        # Check if we're handling a batch or a single item
        if isinstance(query, str) and isinstance(context, str) and isinstance(formatted_prompts, str):
            return self._event_loop.run_until_complete(
                self._generate_async(
                    prompt=formatted_prompts,
                    query=query,
                    context=context,
                    timeout=self._timeout,
                )
            )

        # Batch processing
        return self._event_loop.run_until_complete(
            self._batch_generate(
                prompts=formatted_prompts,
                queries=query,
                contexts=context,
            )
        )

    def get_evals(self) -> list[Eval]:
        """
        Get the list of [Evals](#class-eval) that this TrustworthyRAG instance checks.

        This method returns a copy of the internal evaluation criteria list (to prevent
        accidental modification of the instance's evaluation criteria). The returned list
        contains all evaluation criteria currently configured for this TrustworthyRAG instance,
        whether they are the default evaluations or custom evaluations provided during initialization.
        To change which Evals are run, instantiate a new TrustworthyRAG instance.

        Returns:
            list[Eval]: A list of [Eval](#class-eval) objects which this TrustworthyRAG instance checks.
        """
        return self._evals.copy()

    async def _batch_generate(
        self,
        prompts: Sequence[str],
        queries: Sequence[str],
        contexts: Sequence[str],
    ) -> list[TrustworthyRAGResponse]:
        """Run a batch of generate operations through TrustworthyRAG. The list returned will have the same length as the input list.

        Args:
            prompts (Sequence[str]): list of prompts to run
            queries (Sequence[str]): list of queries corresponding to each prompt
            contexts (Sequence[str]): list of contexts corresponding to each prompt

        Returns:
            list[TrustworthyRAGResponse]: TrustworthyRAG responses/scores for each input (in supplied order)
        """

        # run batch of TrustworthyRAG generate
        rag_responses = await self._batch_async(
            [
                self._generate_async(
                    prompt=prompt,
                    query=query,
                    context=context,
                    timeout=self._timeout,
                    capture_exceptions=True,
                    batch_index=batch_index,
                )
                for batch_index, (prompt, query, context) in enumerate(zip(prompts, queries, contexts))
            ]
        )

        return cast(list[TrustworthyRAGResponse], rag_responses)

    async def _batch_score(
        self,
        responses: Sequence[dict[str, Any]],
        prompts: Sequence[str],
        queries: Sequence[str],
        contexts: Sequence[str],
    ) -> list[TrustworthyRAGScore]:
        """Run a batch of score operations through TrustworthyRAG. The list returned will have the same length as the input list.

        Args:
            responses (Sequence[dict[str, Any]]): list of processed responses to score
            prompts (Sequence[str]): list of prompts corresponding to each response
            queries (Sequence[str]): list of queries corresponding to each response
            contexts (Sequence[str]): list of contexts corresponding to each response

        Returns:
            list[TrustworthyRAGScore]: TrustworthyRAG scores for each input (in supplied order)
        """

        # run batch of TrustworthyRAG score
        rag_scores = await self._batch_async(
            [
                self._score_async(
                    response=response,
                    prompt=prompt,
                    query=query,
                    context=context,
                    timeout=self._timeout,
                    capture_exceptions=True,
                    batch_index=batch_index,
                )
                for batch_index, (response, prompt, query, context) in enumerate(
                    zip(responses, prompts, queries, contexts)
                )
            ]
        )

        return cast(list[TrustworthyRAGScore], rag_scores)

    async def _batch_async(
        self,
        rag_coroutines: Sequence[Coroutine[None, None, Union[TrustworthyRAGResponse, TrustworthyRAGScore]]],
    ) -> Sequence[Union[TrustworthyRAGResponse, TrustworthyRAGScore]]:
        """Runs batch of TrustworthyRAG operations.

        Args:
            rag_coroutines (Sequence[Coroutine[None, None, Union[TrustworthyRAGResponse, TrustworthyRAGScore]]]):
                list of coroutines to run, returning TrustworthyRAGResponse or TrustworthyRAGScore

        Returns:
            Sequence[Union[TrustworthyRAGResponse, TrustworthyRAGScore]]: list of coroutine results, with preserved order
        """
        rag_query_tasks = [asyncio.create_task(rag_coro) for rag_coro in rag_coroutines]

        if self._verbose:
            print(
                "If this progress bar appears frozen, TrustworthyRAG is still processing your dataset so just continue waiting."
            )

            gather_task = tqdm_asyncio.gather(
                *rag_query_tasks,
                total=len(rag_query_tasks),
                desc="Querying TrustworthyRAG...",
                bar_format="{desc} {percentage:3.0f}%|{bar}|",
            )
        else:
            gather_task = asyncio.gather(*rag_query_tasks)  # type: ignore[assignment]

        return cast(
            Sequence[Union[TrustworthyRAGResponse, TrustworthyRAGScore]],
            await gather_task,
        )

    @handle_tlm_exceptions("TrustworthyRAGResponse")
    async def _generate_async(
        self,
        *,
        prompt: str,
        query: str,
        context: str,
        timeout: Optional[float] = None,
        capture_exceptions: bool = False,  # noqa: ARG002
        batch_index: Optional[int] = None,
    ) -> TrustworthyRAGResponse:
        """
        Private asynchronous method to generate a response and evaluation scores.

        Args:
            prompt (str): The formatted prompt for the TLM. If a sequence was provided to the public method,
                it has been processed in the generate method to extract the first element.
            query (str): The user's query string that will be evaluated as part of the RAG system.
            context (str): The context/retrieved documents string that will be evaluated as part of the RAG system.
            timeout (float, optional): Timeout (in seconds) for the API call. If None, no timeout is applied.
            capture_exceptions (bool, optional): Whether to capture exceptions rather than propagating them.
            batch_index (int, optional): Index in the batch for error reporting.

        Returns:
            TrustworthyRAGResponse: A [TrustworthyRAGResponse](#class-trustworthyragresponse) object containing
                the generated response text and evaluation scores for various quality metrics.
        """
        return cast(
            TrustworthyRAGResponse,
            await asyncio.wait_for(
                api.tlm_rag_generate(
                    api_key=self._api_key,
                    prompt=prompt,
                    query=query,
                    context=context,
                    evals=self._evals,
                    quality_preset=self._quality_preset,
                    options=self._options,
                    rate_handler=self._rate_handler,
                    batch_index=batch_index,
                    retries=_TLM_MAX_RETRIES,
                ),
                timeout=timeout,
            ),
        )

    @handle_tlm_exceptions("TrustworthyRAGScore")
    async def _score_async(
        self,
        *,
        response: dict[str, Any],
        prompt: str,
        query: str,
        context: str,
        timeout: Optional[float] = None,
        capture_exceptions: bool = False,  # noqa: ARG002
        batch_index: Optional[int] = None,
    ) -> TrustworthyRAGScore:
        """
        Private asynchronous method to obtain evaluation scores for an existing response.

        Args:
            response (dict[str, Any]): The processed response to evaluate. This is a dictionary containing
                the response text and any additional metadata needed for evaluation.
            prompt (str): The formatted prompt for the TLM. If a sequence was provided to the public method,
                it has been processed in the score method to extract the first element.
            query (str): The user's query string that was used to generate the response.
            context (str): The context/retrieved documents string that was used to generate the response.
            timeout (float, optional): Timeout (in seconds) for the API call. If None, no timeout is applied.
            capture_exceptions (bool, optional): Whether to capture exceptions rather than propagating them.
            batch_index (int, optional): Index in the batch for error reporting.

        Returns:
            TrustworthyRAGScore: A [TrustworthyRAGScore](#class-trustworthyragscore) object containing
                evaluation scores for various quality metrics without generating a new response.
        """
        return cast(
            TrustworthyRAGScore,
            await asyncio.wait_for(
                api.tlm_rag_score(
                    api_key=self._api_key,
                    response=response,
                    prompt=prompt,
                    query=query,
                    context=context,
                    quality_preset=self._quality_preset,
                    options=self._options,
                    rate_handler=self._rate_handler,
                    evals=self._evals,
                    batch_index=batch_index,
                    retries=_TLM_MAX_RETRIES,
                ),
                timeout=timeout,
            ),
        )

    @staticmethod
    def _default_prompt_formatter(query: str, context: str) -> str:
        """
        Format a standard RAG prompt using the provided query and context.

        This method creates a formatted prompt string suitable for RAG systems by combining
        the provided query and context in a structured format. The resulting prompt follows
        a common pattern for RAG application's with clear delimiters between context and query.

        Note: This default formatter does not include a system prompt. If you need to include
        a system prompt or other special instructions to the LLM, use a custom `form_prompt()` function when
        calling `generate()` or `score()`.
        For conversational (multi-turn dialogues rather than single-turn Q&A) RAG apps, we recommend specifying `form_prompt()`.

        Args:
            query (str): The user's question or request to be answered by the RAG system.
            context (str): Context/documents that the RAG system retrieved to help answer the `query`.

        Returns:
            str: A formatted prompt string ready to be sent to the model, with the following structure:
                - Context section with clear delimiters
                - Instructions to use context
                - User query prefixed with "User: "
                - Assistant response starter
        """
        # Start with prompt parts
        prompt_parts = []

        # Add context
        prompt_parts.append("Context information is below.\n")
        prompt_parts.append("---------------------\n")
        prompt_parts.append(f"{context.strip()}\n")
        prompt_parts.append("---------------------\n")

        # Add instruction to use context
        prompt_parts.append("Using the context information provided above, please answer the following question:\n")

        # Add user query
        prompt_parts.append(f"User: {query.strip()}\n")

        # Add assistant response starter
        prompt_parts.append("Assistant: ")

        return "\n".join(prompt_parts)


class Eval:
    """
    Class representing an evaluation for TrustworthyRAG.

    Args:
        name (str): The name of the evaluation, used to identify this specific evaluation in the results.
        criteria (str): The evaluation criteria text that describes what aspect is being evaluated and how.
        query_identifier (str, optional): The exact string used in your evaluation `criteria` to reference the user's query.
            For example, specifying `query_identifier` as "User Question" means your `criteria` should refer to the query as "User Question".
            Leave this value as None (the default) if this Eval doesn't consider the query.
        context_identifier (str, optional): The exact string used in your evaluation `criteria` to reference the retrieved context.
            For example, specifying `context_identifier` as "Retrieved Documents" means your `criteria` should refer to the context as "Retrieved Documents".
            Leave this value as None (the default) if this Eval doesn't consider the context.
        response_identifier (str, optional): The exact string used in your evaluation `criteria` to reference the RAG/LLM response.
            For example, specifying `response_identifier` as "AI Answer" means your `criteria` should refer to the response as "AI Answer".
            Leave this value as None (the default) if this Eval doesn't consider the response.
    """

    def __init__(
        self,
        name: str,
        criteria: str,
        query_identifier: Optional[str] = None,
        context_identifier: Optional[str] = None,
        response_identifier: Optional[str] = None,
    ):
        """
        lazydocs: ignore
        """
        # Validate that at least one identifier is specified
        if query_identifier is None and context_identifier is None and response_identifier is None:
            raise ValueError(
                "At least one of query_identifier, context_identifier, or response_identifier must be specified."
            )

        self.name = name
        self.criteria = criteria
        self.query_identifier = query_identifier
        self.context_identifier = context_identifier
        self.response_identifier = response_identifier

    def __repr__(self) -> str:
        """
        Return a string representation of the Eval object in dictionary format.

        Returns:
            str: A dictionary-like string representation of the Eval object.
        """
        return (
            f"{{\n"
            f"    'name': '{self.name}',\n"
            f"    'criteria': '{self.criteria}',\n"
            f"    'query_identifier': {self.query_identifier!r},\n"
            f"    'context_identifier': {self.context_identifier!r},\n"
            f"    'response_identifier': {self.response_identifier!r}\n"
            f"}}"
        )


_DEFAULT_EVALS: list[dict[str, Optional[str]]] = [
    {
        "name": "context_sufficiency",
        "criteria": "Determine if the Document contains 100% of the information needed to answer the Question. If any external knowledge or assumptions are required, it does not meet the criteria. Each Question component must have explicit support in the Document.",
        "query_identifier": "Question",
        "context_identifier": "Document",
        "response_identifier": None,
    },
    {
        "name": "response_groundedness",
        "criteria": "Review the Response to the Query and assess whether every factual claim in the Response is explicitly supported by the provided Context. A Response meets the criteria if all information is directly backed by evidence in the Context, without relying on assumptions, external knowledge, or unstated inferences. The focus is on whether the Response is fully grounded in the Context, rather than whether it fully addresses the Query. If any claim in the Response lacks direct support or introduces information not present in the Context, the Response is bad and does not meet the criteria.",
        "query_identifier": "Query",
        "context_identifier": "Context",
        "response_identifier": "Response",
    },
    {
        "name": "response_helpfulness",
        "criteria": """Assess whether the AI Assistant Response is a helpful answer to the User Query.
A Response is considered helpful if it makes a genuine attempt to answer the question, even if the answer is incorrect or incomplete. Factual inaccuracies should not affect the assessment. The only thing that matters is whether the Assistant tries to answer the question.
A Response is considered not helpful if it avoids answering the question. For example, by saying or implying things like "I don't know", "Sorry", "No information available", or any other form of refusal or deflection.""",
        "query_identifier": "User Query",
        "context_identifier": None,
        "response_identifier": "AI Assistant Response",
    },
    {
        "name": "query_ease",
        "criteria": """Determine whether the above User Request appears simple and straightforward.
A bad User Request will appear either: ambiguous in intent, complex, purposefully tricky, abnormal, or disgruntled.
A good User Request is phrased clearly and expresses an achievable intent. Basic conversational and non-propositional statements are also considered good.
Should an AI Assistant be able to properly answer the User Request, it is considered good. The AI Assistant handling this User Request has additional knowledge about: the user, domain-specific terms and abbreviations, and any necessary factual information. So a User Request missing information could still be good; vagueness due to undefined pronouns/terms or references to unknown context does not make a User Request bad.
""",
        "query_identifier": "User Request",
        "context_identifier": None,
        "response_identifier": None,
    },
]


def get_default_evals() -> list[Eval]:
    """
    Get the evaluation criteria that are run in TrustworthyRAG by default.

    Returns:
        list[Eval]: A list of [Eval](#class-eval) objects based on pre-configured criteria
        that can be used with TrustworthyRAG.

    Example:
        ```python
        default_evaluations = get_default_evals()

        # You can modify the default Evals by:
        # 1. Adding new evaluation criteria
        # 2. Updating existing criteria with custom text
        # 3. Removing specific evaluations you don't need

        # Run TrustworthyRAG with your modified Evals
        trustworthy_rag = TrustworthyRAG(evals=modified_evaluations)
        ```
    """
    return [
        Eval(
            name=str(eval_config["name"]),
            criteria=str(eval_config["criteria"]),
            query_identifier=eval_config.get("query_identifier"),
            context_identifier=eval_config.get("context_identifier"),
            response_identifier=eval_config.get("response_identifier"),
        )
        for eval_config in _DEFAULT_EVALS
    ]


# Define the response types first
class EvalMetric(TypedDict):
    """Evaluation metric reporting a quality score and optional logs.

    Attributes:
        score (float, optional): score between 0-1 corresponding to the evaluation metric.
        A higher score indicates a higher rating for the specific evaluation criteria being measured.

        log (dict, optional): additional logs and metadata, reported only if the `log` key was specified in [TLMOptions](/tlm/api/python/tlm/#class-tlmoptions).
    """

    score: Optional[float]
    log: NotRequired[dict[str, Any]]


class TrustworthyRAGResponse(dict[str, Union[Optional[str], EvalMetric]]):
    """Object returned by `TrustworthyRAG.generate()` containing generated text and evaluation scores. This class is a dictionary with specific keys.

    Attributes:
        response (str): The generated response text.
        trustworthiness ([EvalMetric](#class-evalmetric)): Overall trustworthiness of the response.
        Additional keys: Various evaluation metrics (context_sufficiency, response_helpfulness, etc.),
            each following the [EvalMetric](#class-evalmetric) structure.

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


# Class for evaluation scores with dynamic evaluation metric keys
class TrustworthyRAGScore(dict[str, EvalMetric]):
    """Object returned by `TrustworthyRAG.score()` containing evaluation scores. This class is a dictionary with specific keys.

    Attributes:
        trustworthiness ([EvalMetric](#class-evalmetric)): Overall trustworthiness of the response.
        Additional keys: Various evaluation metrics (context_sufficiency, response_helpfulness, etc.),
            each following the [EvalMetric](#class-evalmetric) structure.

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

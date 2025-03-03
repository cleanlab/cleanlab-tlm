"""
A version of the Trustworthy Language Model, specifically for real-time validation of RAG responses (Trust Augmented Generation, or TAG).
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Optional

from cleanlab_tlm.errors import (
    MissingApiKeyError,
    ValidationError,
)
from cleanlab_tlm.internal.api.api import tlm_rag_generate, tlm_rag_score
from cleanlab_tlm.internal.concurrency import TlmRateHandler
from cleanlab_tlm.tlm import TLMOptions, TLMResponse, TLMScore, is_notebook

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
        prompt=None,
        query=None,
        context=None,
        system_prompt=None,
        fallback_response=None,
        form_prompt = None
    ) -> TrustworthyRAGResponse:
        """ Required is either specifying at least: query and context, or prompt. 
           default_prompt_format = function that takes in {query, context, system_prompt, fallback_response} 
           returns a standard RAG prompt like we use for Agility. 
        """
        # Use the class method as default if form_prompt is not provided
        if form_prompt is None:
            form_prompt = self._default_prompt_formatter

        # Validate that either prompt or both query and context are provided
        if prompt is None and (query is None or context is None):
            raise ValidationError("Either 'prompt' or both 'query' and 'context' must be provided")

        if prompt is None:
            prompt = form_prompt(query, context, system_prompt, fallback_response)

        # Validate inputs for evaluations (excluding response which isn't available yet)
        is_valid, error_message = self._validate_inputs(
            query=query,
            context=context,
            fallback_response=fallback_response
        )
        if not is_valid:
            raise ValidationError(error_message)

        response = self._event_loop.run_until_complete(
            self._generate_async(
                prompt=prompt,
                query=query,
                context=context,
                system_prompt=system_prompt,
                fallback_response=fallback_response,
            )
        )

        # Cast the response to TrustworthyRAGResponse
        return TrustworthyRAGResponse(**response)

    def score(
        self,
        *,
        response: str,
        prompt=None,
        query=None,
        context=None,
        system_prompt=None,
        fallback_response=None,
        form_prompt = None
    ) -> TrustworthyRAGScore:
        # Use the class method as default if form_prompt is not provided
        if form_prompt is None:
            form_prompt = self._default_prompt_formatter

        # Validate that response is provided
        if response is None or response.strip() == "":
            raise ValidationError("'response' is required and cannot be empty")

        # Validate that either prompt or both query and context are provided
        if prompt is None and (query is None or context is None):
            raise ValidationError("Either 'prompt' or both 'query' and 'context' must be provided")

        # Form the prompt if not provided
        if prompt is None:
            prompt = form_prompt(query, context, system_prompt, fallback_response)

        # Validate inputs for evaluations
        is_valid, error_message = self._validate_inputs(
            query=query,
            context=context,
            response=response,
            fallback_response=fallback_response
        )
        if not is_valid:
            raise ValidationError(error_message)

        score_response = self._event_loop.run_until_complete(
            self._score_async(
                response=response,
                prompt=prompt,
                query=query,
                context=context,
                system_prompt=system_prompt,
                fallback_response=fallback_response,
            )
        )

        # Cast the response to TrustworthyRAGScore
        return TrustworthyRAGScore(**score_response)

    def get_evals(self) -> list[Eval]:
        """
        Returns a copy of the evaluation criteria currently being used by this TrustworthyRAG instance.
        
        Returns:
            A list of Eval objects with the current evaluation criteria
        """
        return self._evals.copy()

    def _validate_inputs(self, query=None, context=None, response=None, fallback_response=None) -> tuple[bool, Optional[str]]:
        """
        Validate that all required inputs for evaluations are present.
        
        Args:
            query: The user query
            context: The context used for RAG
            response: The response to evaluate
            fallback_response: The fallback response
            
        Returns:
            A tuple of (is_valid, error_message)
            - is_valid: True if all required inputs are present, False otherwise
            - error_message: Error message if validation fails, None otherwise
        """
        inputs_data = {
            "query": query,
            "context": context,
            "response": response,
            "fallback_response": fallback_response
        }

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
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        fallback_response: Optional[str] = None
    ) -> str:
        """
        Format a standard RAG prompt using the provided components.
        
        Args:
            query: The user's question or request
            context: Retrieved context/documents to help answer the query
            system_prompt: Optional system instructions for the model
            fallback_response: Optional fallback response to use if context is insufficient
            
        Returns:
            A formatted prompt string ready to be sent to the model
        """
        # Start with system prompt if provided
        prompt_parts = []

        if system_prompt:
            prompt_parts.append(f"{system_prompt.strip()}\n")

        # Add context
        prompt_parts.append("Context information is below.\n")
        prompt_parts.append("---------------------\n")
        prompt_parts.append(f"{context.strip()}\n")
        prompt_parts.append("---------------------\n")

        # Add fallback response guidance if provided
        if fallback_response:
            prompt_parts.append(
                "Note: If the context above doesn't provide sufficient information to answer "
                "the user's question, you may use the following fallback response as a guide:\n"
            )
            prompt_parts.append(f"{fallback_response.strip()}\n")

        # Add user query
        prompt_parts.append(f"User: {query.strip()}\n")

        # Add assistant response starter
        prompt_parts.append("Assistant: ")

        return "\n".join(prompt_parts)

    # @handle_tlm_exceptions("TLMResponse")
    async def _generate_async(
        self,
        *,
        prompt: str,
        query: Optional[str] = None,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        fallback_response: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> TLMResponse:
        """
        Private asynchronous method to get response and trustworthiness score from TLM.

        Args:
            prompt (str): prompt for the TLM
            client_session (aiohttp.ClientSession, optional): async HTTP session to use for TLM query. Defaults to None (creates a new session).
            timeout: timeout (in seconds) to run the prompt, defaults to None (no timeout)
            capture_exceptions (bool): if True, the returned [TLMResponse](#class-tlmresponse) object will include error details and retry information if any errors or timeouts occur during processing.
            batch_index: index of the prompt in the batch, used for error messages
            constrain_outputs: list of strings to constrain the output of the TLM to
        Returns:
            TLMResponse: [TLMResponse](#class-tlmresponse) object containing the response and trustworthiness score.
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
                system_prompt=system_prompt,
                fallback_response=fallback_response,
                evals=self._evals,
            ),
            timeout=timeout,
        )

        return response_json

    # @handle_tlm_exceptions("TLMResponse")
    async def _score_async(
        self,
        *,
        response: str,
        prompt: str,
        query: Optional[str] = None,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        fallback_response: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> TLMResponse:
        """
        Private asynchronous method to get trustworthiness score from TLM.

        Args:
            response (str): response to score
            prompt (str): prompt for the TLM
            client_session (aiohttp.ClientSession, optional): async HTTP session to use for TLM query. Defaults to None (creates a new session).
            timeout: timeout (in seconds) to run the prompt, defaults to None (no timeout)
            capture_exceptions (bool): if True, the returned [TLMResponse](#class-tlmresponse) object will include error details and retry information if any errors or timeouts occur during processing.
            batch_index: index of the prompt in the batch, used for error messages
        Returns:
            TLMResponse: [TLMResponse](#class-tlmresponse) object containing the trustworthiness score.
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
                system_prompt=system_prompt,
                fallback_response=fallback_response,
                evals=self._evals,
            ),
            timeout=timeout,
        )

        return response_json



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


class TrustworthyRAGResponse(TLMResponse):
    """
    A typed dict similar to [TLMResponse](../tlm/#class-tlmresponse) but containing an extra key `calibrated_score`.
    View [TLMResponse](../tlm/#class-tlmresponse) for the description of the other keys in this dict.

    Attributes:
        calibrated_score (float, optional): score between 0 and 1 that has been calibrated to the provided ratings.
        A higher score indicates a higher confidence that the response is correct/trustworthy.
    """

    rag_eval_scores: Optional[dict[str, float]]


class TrustworthyRAGScore(TLMScore):
    """
    A typed dict similar to [TLMScore](../tlm/#class-tlmscore) but containing an extra key `calibrated_score`.
    View [TLMScore](../tlm/#class-tlmscore) for the description of the other keys in this dict.

    Attributes:
        calibrated_score (float, optional): score between 0 and 1 that has been calibrated to the provided ratings.
        A higher score indicates a higher confidence that the response is correct/trustworthy.
    """

    rag_eval_scores: Optional[dict[str, float]]


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


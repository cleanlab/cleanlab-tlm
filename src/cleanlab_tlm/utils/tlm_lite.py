"""
TLM Lite is a version of the [Trustworthy Language Model (TLM)](../tlm) that enables the use of different LLMs for generating the response and for scoring its trustworthiness.
"""

import os
import warnings
from collections.abc import Sequence
from typing import Optional, Union, cast

import numpy as np

from cleanlab_tlm.errors import MissingApiKeyError, ValidationError
from cleanlab_tlm.internal.types import TLMQualityPreset
from cleanlab_tlm.internal.validation import (
    get_tlm_lite_response_options,
    validate_tlm_lite_score_options,
)
from cleanlab_tlm.tlm import (
    TLM,
    TLMOptions,
    TLMResponse,
)


class TLMLite:
    """
    A version of the Trustworthy Language Model (TLM) that enables the use of different LLMs for generating the response and for scoring its trustworthiness.

    TLMLite should be used if you want to use a better model to generate responses but want to get cheaper and quicker trustworthiness score
    evaluations by using smaller models.

    Possible arguments for `TLMLite()` are documented below. Most of the input arguments for this class are similar to those for TLM, major differences will be described below.

    Args:
        response_model (str): LLM used to produce the response to the given prompt.
            Do not specify the model to use for scoring  trustworthiness here, instead specify that model in the `options` argument.
            The list of supported model strings can be found in the [TLMOptions](../tlm/#class-tlmoptions) documentation, by default, the model is "gpt-4o".

        quality_preset (TLMQualityPreset, default = "medium"): preset configuration to control the quality of TLM trustworthiness scores vs. runtimes/costs.
            This preset only applies to the model computing the trustworthiness score.
            Supported options are only "medium" or "low", because TLMLite is not intended to improve response accuracy
            (use the regular [TLM](../tlm) for that).

        options ([TLMOptions](../tlm/#class-tlmoptions), optional): a typed dict of advanced configuration options.
            Most of these options only apply to the model scoring  trustworthiness, except for "max_tokens", which applies to the response model as well.
            Specify which model to use for scoring trustworthiness in these options.
            For more details about the options, see the documentation for [TLMOptions](../tlm/#class-tlmoptions).

        timeout (float, optional): timeout (in seconds) to apply to each TLM prompt.

        verbose (bool, optional): whether to print outputs during execution, i.e., whether to show a progress bar when TLM is prompted with batches of data.
    """

    def __init__(
        self,
        response_model: str = "gpt-4o",
        quality_preset: TLMQualityPreset = "medium",
        *,
        api_key: Optional[str] = None,
        options: Optional[TLMOptions] = None,
        timeout: Optional[float] = None,
        verbose: Optional[bool] = None,
    ) -> None:
        """
        lazydocs: ignore
        """
        self._api_key = api_key or os.environ.get("CLEANLAB_TLM_API_KEY")
        if self._api_key is None:
            raise MissingApiKeyError
        self._response_model = response_model

        if quality_preset not in {"low", "medium"}:
            raise ValidationError(
                f"Invalid quality preset: {quality_preset}. TLMLite only supports 'low' and 'medium' presets."
            )
        self._score_quality_preset = quality_preset

        if options is not None:
            validate_tlm_lite_score_options(options)
        self._score_options = options
        self._response_options = cast(
            TLMOptions,
            get_tlm_lite_response_options(self._score_options, self._response_model),
        )

        self._timeout = timeout if timeout is not None and timeout > 0 else None  # TODO: better timeout handling
        self._verbose = verbose

        self._tlm_response = TLM(
            quality_preset="base",
            api_key=self._api_key,
            options=self._response_options,
            timeout=self._timeout,
            verbose=self._verbose,
        )

        self._tlm_score = TLM(
            quality_preset=self._score_quality_preset,
            api_key=self._api_key,
            options=self._score_options,
            timeout=self._timeout,
            verbose=self._verbose,
        )

    def prompt(
        self,
        prompt: Union[str, Sequence[str]],
    ) -> Union[TLMResponse, list[TLMResponse]]:
        """
        Similar to [`TLM.prompt()`](../tlm/#method-prompt), view documentation there for expected input arguments and outputs.
        """
        prompt_response = self._tlm_response.prompt(prompt)

        # single call
        if isinstance(prompt, str) and isinstance(prompt_response, dict):
            if prompt_response["response"] is None:
                return prompt_response

            return self._score(
                prompt,
                prompt_response["response"],
                perplexity=prompt_response["log"]["perplexity"],
            )

        # batch call
        if isinstance(prompt, Sequence) and isinstance(prompt_response, list):
            prompt_succeeded_mask = np.array([res["response"] is not None for res in prompt_response])

            if not np.any(prompt_succeeded_mask):  # all prompts failed
                return prompt_response

            # handle masking with numpy for easier indexing
            prompt_succeeded = np.array(prompt)[prompt_succeeded_mask].tolist()
            prompt_response_succeeded = np.array(prompt_response)[prompt_succeeded_mask]

            response_succeeded = [r["response"] for r in prompt_response_succeeded]
            perplexity_succeeded = [r["log"]["perplexity"] for r in prompt_response_succeeded]
            score_response_succeeded = self._batch_score(prompt_succeeded, response_succeeded, perplexity_succeeded)

            tlm_response = np.array(prompt_response)
            tlm_response[prompt_succeeded_mask] = np.array(score_response_succeeded)

            return cast(list[TLMResponse], tlm_response.tolist())

        raise ValueError("prompt and prompt_response do not have matching types")

    def try_prompt(
        self,
        prompt: Sequence[str],
    ) -> list[TLMResponse]:
        """
        lazydocs: ignore
        Deprecated method. Use `prompt()` instead.
        """
        warnings.warn(
            "Deprecated method. Use `prompt()` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cast(list[TLMResponse], self.prompt(prompt))

    def _score(
        self,
        prompt: str,
        response: str,
        perplexity: Optional[float],
    ) -> TLMResponse:
        """
        Private method to get trustworthiness score for a single example and process the outputs into a TLMResponse dictionary.

        Args:
            prompt: prompt for the TLM to evaluate
            response: response corresponding to the input prompt
            perplexity: perplexity of the response (given by LLM)
        Returns:
            TLMResponse: [TLMResponse](#class-tlmresponse) object containing the response and trustworthiness score
        """
        score_response = self._tlm_score.get_trustworthiness_score(prompt, response, perplexity=perplexity)

        if not isinstance(score_response, dict):
            raise TypeError(f"score_response has invalid type {type(score_response)}")

        return {"response": response, **score_response}

    def _batch_score(
        self,
        prompt: Sequence[str],
        response: Sequence[str],
        perplexity: Sequence[Optional[float]],
    ) -> list[TLMResponse]:
        """
        Private method to get trustworthiness score for a batch of examples and process the outputs into TLMResponse dictionaries,
        handling any failures (errors of timeouts) by returning None in place of the failures.

        Args:
            prompt: list of prompts for the TLM to evaluate
            response: list of responses corresponding to the input prompt
            perplexity: list of perplexity scores of the response (given by LLM)
        Returns:
            list[TLMResponse]: list of [TLMResponse](#class-tlmresponse) object containing the response and trustworthiness score.
                In case of any failures, the return list will contain None in place of the TLM response for that example.
        """
        score_response = self._tlm_score.try_get_trustworthiness_score(prompt, response, perplexity=perplexity)

        assert len(prompt) == len(score_response)

        if not all(isinstance(score, dict) for score in score_response):
            raise TypeError("score_response has invalid type")

        return [{"response": res, **score} if score else None for res, score in zip(response, score_response)]

    def get_model_names(self) -> dict[str, str]:
        """Returns the underlying LLMs used to generate responses and score their trustworthiness."""
        return {
            "response_model": self._tlm_response.get_model_name(),
            "score_model": self._tlm_score.get_model_name(),
        }

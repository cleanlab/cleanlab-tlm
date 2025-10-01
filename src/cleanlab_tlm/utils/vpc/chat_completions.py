"""
Real-time evaluation of responses from OpenAI Chat Completions API.

If you are using OpenAI's Chat Completions API, this module allows you to incorporate TLM trust scoring without any change to your existing code.
It works for any OpenAI LLM model, as well as the many other non-OpenAI LLMs that are also usable via Chat Completions API (Gemini, DeepSeek, Llama, etc).

This module is specifically for VPC users of TLM, the BASE_URL environment variable must be set to the VPC endpoint.
If you are not using VPC, use the `cleanlab_tlm.utils.chat_completions` module instead.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional, Union, cast

import requests

from cleanlab_tlm.internal.base import BaseTLM
from cleanlab_tlm.internal.constants import _VALID_TLM_QUALITY_PRESETS_CHAT_COMPLETIONS
from cleanlab_tlm.tlm import TLMScore
from cleanlab_tlm.utils.per_field_score_utils import _get_untrustworthy_fields
from cleanlab_tlm.utils.vpc.tlm import VPCTLMOptions

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion


class TLMChatCompletion(BaseTLM):
    """
    Represents a Trustworthy Language Model (TLM) instance specifically designed for evaluating OpenAI Chat Completions responses.

    This class provides a TLM wrapper that can be used to evaluate the quality and trustworthiness of responses from any OpenAI model
    by passing in the inputs to OpenAI's Chat Completions API and the ChatCompletion response object.

    This module is specifically for VPC users of TLM, the BASE_URL environment variable must be set to the VPC endpoint.
    If you are not using VPC, use the `cleanlab_tlm.utils.chat_completions` module instead.

    Args:
        quality_preset ({"base", "low", "medium"}, default = "medium"): an optional preset configuration to control
            the quality of TLM trustworthiness scores vs. latency/costs.

        options ([TLMOptions](#class-tlmoptions), optional): a typed dict of configurations you can optionally specify.
            See detailed documentation under [TLMOptions](#class-tlmoptions).

        timeout (float, optional): timeout (in seconds) to apply to each TLM evaluation.
    """

    def __init__(
        self,
        quality_preset: str = "medium",
        *,
        options: Optional[VPCTLMOptions] = None,
        timeout: Optional[float] = None,
        request_headers: Optional[dict[str, str]] = None,
    ):
        """
        lazydocs: ignore
        """
        super().__init__(
            quality_preset=quality_preset,
            valid_quality_presets=_VALID_TLM_QUALITY_PRESETS_CHAT_COMPLETIONS,
            support_custom_eval_criteria=True,
            api_key=".",
            options=options,
            timeout=timeout,
            verbose=False,
            allow_custom_model=True,
            valid_options_keys=set(VPCTLMOptions.__annotations__.keys()),
        )
        self._request_headers = request_headers or {}

    def score(
        self,
        *,
        response: ChatCompletion,
        **openai_kwargs: Any,
    ) -> TLMScore:
        """Score the trustworthiness of an OpenAI ChatCompletion response.

        Args:
            response (ChatCompletion): The OpenAI ChatCompletion response object to evaluate
            **openai_kwargs (Any): The original kwargs passed to OpenAI's create() method, must include 'messages'

        Returns:
            TLMScore: A dict containing the trustworthiness score and optional logs
        """
        try:
            from openai.lib._parsing._completions import type_to_response_format_param
        except ImportError as e:
            raise ImportError(
                f"OpenAI is required to use the {self.__class__.__name__} class. Please install it with `pip install openai`."
            ) from e

        if (base_url := os.environ.get("BASE_URL")) is None:
            raise ValueError("BASE_URL is not set. Please set it in the environment variables.")

        # replace the model used for scoring with the specified model in options
        openai_kwargs["model"] = self._options["model"]

        if "response_format" in openai_kwargs:
            openai_kwargs["response_format"] = type_to_response_format_param(openai_kwargs["response_format"])

        res = requests.post(
            f"{base_url}/chat/score",
            json={
                "quality_preset": self._quality_preset,
                "options": self._options,
                "completion": response.model_dump(),
                **openai_kwargs,
            },
            timeout=self._timeout,
            headers=self._request_headers,
        )

        if not res.ok:
            try:
                res.raise_for_status()
            except requests.exceptions.HTTPError as e:
                raise requests.exceptions.HTTPError(
                    f"TLM score API error: {e.response.status_code} {e.response.reason} - {e}"
                ) from e

        res_json = res.json()
        tlm_result = {"trustworthiness_score": res_json["tlm_metadata"]["trustworthiness_score"]}

        if self._return_log:
            log = {}

            log_options = cast(list[str], self._options.get("log", []))
            if "explanation" in log_options:
                explanation = res_json["tlm_metadata"].get("log", {}).get("explanation")
                log["explanation"] = explanation

            if "per_field_score" in log_options:
                per_field_score = res_json["tlm_metadata"].get("log", {}).get("per_field_score")
                log["per_field_score"] = per_field_score

            tlm_result["log"] = log

        return cast(TLMScore, tlm_result)

    def get_untrustworthy_fields(
        self,
        *,
        response: Optional[ChatCompletion] = None,
        tlm_result: Union[TLMScore, ChatCompletion],
        threshold: float = 0.8,
        display_details: bool = True,
    ) -> list[str]:
        """Gets the fields of a structured output response that are considered untrustworthy by TLM.
        Only works for responses that are valid JSON objects (uses `response_format` to specify the output format).
        Prints detailed information about the untrustworthy fields if `display_details` is True.

        Args:
            response (ChatCompletion): The OpenAI ChatCompletion response object to evaluate
            tlm_result (TLMScore | ChatCompletion): The result object from a previous TLM call
            threshold (float): The threshold for considering a field untrustworthy
            display_details (bool): Whether to display detailed information about the untrustworthy fields

        Returns:
            list[str]: The fields of the response that are considered untrustworthy by TLM
        """

        return _get_untrustworthy_fields(
            response=response,
            tlm_result=tlm_result,
            threshold=threshold,
            display_details=display_details,
            class_name=self.__class__.__name__,
        )

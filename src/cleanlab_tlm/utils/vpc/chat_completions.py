"""
Real-time evaluation of responses from OpenAI Chat Completions API.

If you are using OpenAI's Chat Completions API, this module allows you to incorporate TLM trust scoring without any change to your existing code.
It works for any OpenAI LLM model, as well as the many other non-OpenAI LLMs that are also usable via Chat Completions API (Gemini, DeepSeek, Llama, etc).

This module is specifically for VPC users of TLM, the BASE_URL environment variable must be set to the VPC endpoint.
If you are not using VPC, use the `cleanlab_tlm.utils.chat_completions` module instead.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Optional, Union, cast

import requests

from cleanlab_tlm.internal.base import BaseTLM
from cleanlab_tlm.internal.constants import _VALID_TLM_QUALITY_PRESETS_CHAT_COMPLETIONS
from cleanlab_tlm.tlm import TLMScore
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

    def get_per_field_score_breakdown(
        self,
        *,
        response: Optional[ChatCompletion] = None,
        tlm_result: Union[TLMScore, ChatCompletion],
        threshold: float = 0.8,
        display_details: bool = True,
    ) -> list[str]:
        """Get the per-field score breakdown for an OpenAI ChatCompletion response."""

        try:
            from openai.types.chat import ChatCompletion
        except ImportError as e:
            raise ImportError(
                f"OpenAI is required to use the {self.__class__.__name__} class. Please install it with `pip install openai`."
            ) from e

        if isinstance(tlm_result, dict):
            if response is None:
                raise ValueError("'response' is required when tlm_result is a TLMScore object")

            tlm_metadata = tlm_result
            response_text = response.choices[0].message.content or "{}"

        elif isinstance(tlm_result, ChatCompletion):
            if getattr(tlm_result, "tlm_metadata", None) is None:
                raise ValueError("tlm_result must contain tlm_metadata.")

            tlm_metadata = tlm_result.tlm_metadata  # type: ignore
            response_text = tlm_result.choices[0].message.content or "{}"

        else:
            raise TypeError("tlm_result must be a TLMScore or ChatCompletion object.")

        if "per_field_score" not in tlm_metadata.get("log", {}):
            raise ValueError("per_field_score is not present in the log")

        so_response = json.loads(response_text)
        per_field_score = tlm_metadata["log"]["per_field_score"]
        per_score_details = []

        for key, value in per_field_score.items():
            score = value["score"]
            if float(score) < threshold:
                key_details = {
                    "response": so_response[key],
                    "score": score,
                    "explanation": value["explanation"],
                }
                per_score_details.append({key: key_details})

        per_score_details.sort(key=lambda x: next(iter(x.values()))["score"])
        untrustworthy_fields = [next(iter(item.keys())) for item in per_score_details]

        if display_details:
            if len(untrustworthy_fields) == 0:
                print("No untrustworthy fields found")

            else:
                print(f"Untrustworthy fields: {untrustworthy_fields}\n")
                for item in per_score_details:
                    print(next(iter(item.keys())))
                    details = next(iter(item.values()))
                    print(f"Response: {details['response']}")
                    print(f"Score: {details['score']}")
                    print(f"Explanation: {details['explanation']}")
                    print()

        return untrustworthy_fields

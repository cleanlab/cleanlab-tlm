"""
Real-time evaluation of responses from OpenAI Chat Completions API.

If you are using OpenAI's Chat Completions API, this module allows you to incorporate TLM trust scoring without any change to your existing code.
It works for any OpenAI LLM model, as well as the many other non-OpenAI LLMs that are also usable via Chat Completions API (Gemini, DeepSeek, Llama, etc).

This module is specifically for VPC users of TLM, the BASE_URL environment variable must be set to the VPC endpoint.
If you are not using VPC, use the `cleanlab_tlm.utils.chat_completions` module instead.
"""

import os
from typing import TYPE_CHECKING, Any, Optional

import requests

from cleanlab_tlm.internal.base import BaseTLM
from cleanlab_tlm.internal.constants import _VALID_TLM_QUALITY_PRESETS_CHAT_COMPLETIONS
from cleanlab_tlm.internal.types import JSONDict
from cleanlab_tlm.tlm import TLMOptions

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
        options: Optional[TLMOptions] = None,
        timeout: Optional[float] = None,
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
        )

    def score(
        self,
        *,
        response: "ChatCompletion",
        **openai_kwargs: Any,
    ) -> JSONDict:
        """Score the trustworthiness of an OpenAI ChatCompletion response.

        Args:
            response (ChatCompletion): The OpenAI ChatCompletion response object to evaluate
            **openai_kwargs (Any): The original kwargs passed to OpenAI's create() method, must include 'messages'

        Returns:
            TLMScore: A dict containing the trustworthiness score and optional logs
        """
        if (base_url := os.environ.get("BASE_URL")) is None:
            raise ValueError("BASE_URL is not set. Please set it in the environment variables.")

        # replace the model used for scoring with the specified model in options
        openai_kwargs["model"] = self._options["model"]

        res = requests.post(
            f"{base_url}/chat/score",
            json={
                "quality_preset": self._quality_preset,
                "options": self._options,
                "completion": response.model_dump(),
                **openai_kwargs,
            },
            timeout=self._timeout,
        )

        res_json = res.json()

        return {"trustworthiness_score": res_json["tlm_metadata"]["trustworthiness_score"]}

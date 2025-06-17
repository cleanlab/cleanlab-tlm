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
    def __init__(
        self,
        quality_preset: str = "medium",
        *,
        options: Optional[TLMOptions] = None,
        timeout: Optional[float] = None,
    ):
        super().__init__(
            quality_preset=quality_preset,
            valid_quality_presets=_VALID_TLM_QUALITY_PRESETS_CHAT_COMPLETIONS,
            support_custom_eval_criteria=True,
            api_key=".",
            options=options,
            timeout=timeout,
            verbose=False,
        )

        self._options["quality_preset"] = self._quality_preset

    def score(
        self,
        *,
        response: "ChatCompletion",
        **openai_kwargs: Any,
    ) -> JSONDict:
        if (base_url := os.environ.get("BASE_URL")) is None:
            raise ValueError("BASE_URL is not set. Please set it in the environment variables.")

        # replace the model used for scoring with the specified model in options
        openai_kwargs["model"] = self._options["model"]

        res = requests.post(
            f"{base_url}/chat/score",
            json={
                "tlm_options": self._options,
                "completion": response.model_dump(),
                **openai_kwargs,
            },
            timeout=self._timeout,
        )

        res_json = res.json()

        return {"trustworthiness_score": res_json["tlm_metadata"]["trustworthiness_score"]}

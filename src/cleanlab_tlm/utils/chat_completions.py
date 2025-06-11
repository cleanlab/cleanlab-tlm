import os
from typing import Any

import requests

from cleanlab_tlm.internal.types import JSONDict

BASE_URL = os.environ.get("BASE_URL")


class TLMChatCompletion:
    try:
        from openai.types.chat import ChatCompletion
    except ImportError:
        raise ImportError(
            "OpenAI is required to use the TLMChatCompletion class. Please install it with `pip install openai`."
        )

    def __init__(self, quality_preset: str = "medium", timeout: float = 60):
        self._tlm_options = {}
        self._tlm_options["quality_preset"] = quality_preset

        self._timeout = timeout  # arbitrary timeout for now

    def create(
        self,
        **openai_kwargs: Any,
    ) -> ChatCompletion:
        from openai.types.chat import ChatCompletion

        if BASE_URL is None:
            raise ValueError("BASE_URL is not set. Please set it in the environment variables.")

        res = requests.post(
            f"{BASE_URL}/completions",
            json={
                "tlm_options": self._tlm_options,
                **openai_kwargs,
            },
            timeout=self._timeout,
        )

        res_json = res.json()

        return ChatCompletion(**res_json)

    def score(
        self,
        response: ChatCompletion,
        **openai_kwargs: Any,
    ) -> JSONDict:
        if BASE_URL is None:
            raise ValueError("BASE_URL is not set. Please set it in the environment variables.")

        res = requests.post(
            f"{BASE_URL}/score",
            json={
                "tlm_options": self._tlm_options,
                "completion": response.model_dump(),
                **openai_kwargs,
            },
            timeout=self._timeout,
        )

        res_json = res.json()

        return {"trustworthiness_score": res_json["tlm_metadata"]["trustworthiness_score"]}

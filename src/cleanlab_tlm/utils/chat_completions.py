import asyncio
from typing import TYPE_CHECKING, Any, Optional, cast

import aiohttp

from cleanlab_tlm.internal.api import api
from cleanlab_tlm.internal.base import BaseTLM
from cleanlab_tlm.internal.constants import (
    _DEFAULT_TLM_QUALITY_PRESET,
    _VALID_TLM_QUALITY_PRESETS_CHAT_COMPLETIONS,
)
from cleanlab_tlm.internal.types import TLMQualityPreset
from cleanlab_tlm.internal.validation_openai import validate_openai_kwargs
from cleanlab_tlm.tlm import TLM, TLMOptions, TLMScore
from cleanlab_tlm.utils.chat import form_prompt_string

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion


class TLMChatCompletion(BaseTLM):

    def __init__(
        self,
        quality_preset: TLMQualityPreset = _DEFAULT_TLM_QUALITY_PRESET,
        *,
        api_key: Optional[str] = None,
        options: Optional[TLMOptions] = None,
        timeout: Optional[float] = None,
    ):
        super().__init__(
            quality_preset=quality_preset,
            valid_quality_presets=_VALID_TLM_QUALITY_PRESETS_CHAT_COMPLETIONS,
            support_custom_eval_criteria=False,
            api_key=api_key,
            options=options,
            timeout=timeout,
            verbose=False,
        )

        self._tlm = TLM(
            quality_preset=quality_preset,
            api_key=api_key,
            options=options,
            timeout=timeout,
        )

    def create(
        self,
        **openai_kwargs: Any,
    ) -> "ChatCompletion":
        validate_openai_kwargs(**openai_kwargs)

        return self._event_loop.run_until_complete(
            self._create_async(openai_kwargs),
        )

    def score(
        self,
        response: "ChatCompletion",
        **openai_kwargs: Any,
    ) -> TLMScore:

        if (messages := openai_kwargs.get("messages")) is None:
            raise ValueError("messages is a required OpenAI input argument.")
        tools = openai_kwargs.get("tools", None)

        prompt_text = form_prompt_string(messages, tools)
        response_text = _get_string_response(response)

        return cast(
            TLMScore, self._tlm.get_trustworthiness_score(prompt_text, response_text)
        )

    async def _create_async(
        self,
        openai_kwargs: dict[str, Any],
        client_session: Optional[aiohttp.ClientSession] = None,
    ) -> "ChatCompletion":
        try:
            from openai.types.chat import ChatCompletion
        except ImportError:
            raise ImportError(
                "OpenAI is required to use the TLMChatCompletion class. Please install it with `pip install openai`."
            )

        response_json = await asyncio.wait_for(
            api.tlm_openai_create(
                api_key=self._api_key,
                quality_preset=self._quality_preset,
                tlm_options=self._format_tlm_options(),
                openai_kwargs=openai_kwargs,
                rate_handler=self._rate_handler,
                client_session=client_session,
            ),
            timeout=self._timeout,
        )

        return ChatCompletion(**response_json)

    def _format_tlm_options(self) -> dict[str, Any]:
        formatted_options = {}
        if (log := self._options.get("log")) is not None:
            formatted_options["log"] = log

        return formatted_options


def _get_string_response(response: "ChatCompletion") -> str:
    try:
        from openai.types.chat import ChatCompletion
    except ImportError:
        raise ImportError(
            "OpenAI is required to use the TLMChatCompletion class. Please install it with `pip install openai`."
        )

    if not isinstance(response, ChatCompletion):
        raise TypeError("The response is not an OpenAI ChatCompletion object.")
    if response.choices[0].message.content is None:
        raise ValueError(
            "The OpenAI ChatCompletion object does not contain a message content."
        )
    return str(response.choices[0].message.content)

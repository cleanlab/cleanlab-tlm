from typing import TYPE_CHECKING, Any, Optional, cast

from cleanlab_tlm.internal.base import BaseTLM
from cleanlab_tlm.internal.constants import (
    _DEFAULT_TLM_QUALITY_PRESET,
    _VALID_TLM_QUALITY_PRESETS_CHAT_COMPLETIONS,
)
from cleanlab_tlm.internal.types import TLMQualityPreset
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
            support_custom_eval_criteria=True,
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

    def score(
        self,
        *,
        response: "ChatCompletion",
        **openai_kwargs: Any,
    ) -> TLMScore:
        if (messages := openai_kwargs.get("messages")) is None:
            raise ValueError("messages is a required OpenAI input argument.")
        tools = openai_kwargs.get("tools", None)

        prompt_text = form_prompt_string(messages, tools)
        response_text = _get_string_response(response)

        return cast(TLMScore, self._tlm.get_trustworthiness_score(prompt_text, response_text))

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
        raise ValueError("The OpenAI ChatCompletion object does not contain a message content.")
    return str(response.choices[0].message.content)

"""
Real-time evaluation of responses from OpenAI Responses API.

If you are using OpenAI's Responses API, this module allows you to incorporate TLM trust scoring without any change to your existing code.
"""

import asyncio
from typing import TYPE_CHECKING, Any, Optional, cast

from cleanlab_tlm.internal.api.api import tlm_chat_completions_score
from cleanlab_tlm.internal.base import BaseTLM
from cleanlab_tlm.internal.constants import (
    _DEFAULT_TLM_QUALITY_PRESET,
    _VALID_TLM_QUALITY_PRESETS,
)
from cleanlab_tlm.internal.types import TLMQualityPreset
from cleanlab_tlm.tlm import TLM, TLMOptions, TLMScore
from cleanlab_tlm.utils.chat import (
    _form_prompt_responses_api,
    form_response_string_responses_api,
)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion
    from openai.types.responses import Response


class TLMResponses(BaseTLM):
    """
    Represents a Trustworthy Language Model (TLM) instance specifically designed for evaluating OpenAI Responses API responses.

    This class provides a TLM wrapper that can be used to evaluate the quality and trustworthiness of responses from any OpenAI model
    by passing in the inputs to OpenAI's Responses API and the Responses object.

    Args:
        quality_preset ({"base", "low", "medium", "high", "best"}, default = "medium"): an optional preset configuration to control
            the quality of TLM trustworthiness scores vs. latency/costs.

        api_key (str, optional): Cleanlab TLM API key. If not provided, will attempt to read from CLEANLAB_API_KEY environment variable.

        options ([TLMOptions](../tlm/#class-tlmoptions), optional): a typed dict of configurations you can optionally specify.
            See detailed documentation under [TLMOptions](../tlm/#class-tlmoptions).

        timeout (float, optional): timeout (in seconds) to apply to each TLM evaluation.
    """

    def __init__(
        self,
        quality_preset: TLMQualityPreset = _DEFAULT_TLM_QUALITY_PRESET,
        *,
        api_key: Optional[str] = None,
        options: Optional[TLMOptions] = None,
        timeout: Optional[float] = None,
    ):
        """
        lazydocs: ignore
        """
        super().__init__(
            quality_preset=quality_preset,
            valid_quality_presets=_VALID_TLM_QUALITY_PRESETS,
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
        response: "Response",
        **openai_kwargs: Any,
    ) -> TLMScore:
        """Score the trustworthiness of an OpenAI Responses API response.

        Args:
            response (Responses): The OpenAI Responses object to evaluate
            **openai_kwargs (Any): The original kwargs passed to OpenAI's responses.create() method

        Returns:
            TLMScore: A dict containing the trustworthiness score and optional logs
        """

        # handle structured outputs differently
        if "text" in openai_kwargs or "text_format" in openai_kwargs:
            converted_chat_completion = _convert_responses_to_chat_completion(response)
            converted_kwargs = _responses_kwargs_to_chat_completion_kwargs(openai_kwargs)
            combined_kwargs = {
                "quality_preset": self._quality_preset,
                **converted_kwargs,
                **self._options,
            }

            return cast(
                TLMScore,
                self._event_loop.run_until_complete(
                    asyncio.wait_for(
                        tlm_chat_completions_score(
                            api_key=self._api_key,
                            response=converted_chat_completion,
                            **combined_kwargs,
                        ),
                        timeout=self._timeout,
                    )
                ),
            )

        # all other cases
        tools = openai_kwargs.get("tools", None)

        prompt_text = _form_prompt_responses_api(openai_kwargs["input"], tools)
        response_text = form_response_string_responses_api(response=response)

        return cast(TLMScore, self._tlm.get_trustworthiness_score(prompt_text, response_text))


def _convert_responses_to_chat_completion(response: "Response") -> "ChatCompletion":
    try:
        from openai.types.chat.chat_completion import (
            ChatCompletion,
            Choice,
            ChoiceLogprobs,
        )
        from openai.types.chat.chat_completion_message import ChatCompletionMessage
        from openai.types.chat.chat_completion_token_logprob import (
            ChatCompletionTokenLogprob,
            TopLogprob,
        )
        from openai.types.completion_usage import CompletionUsage

    except ImportError as e:
        raise ImportError(
            "OpenAI is required to use the TLMResponses class. Please install it with `pip install openai`."
        ) from e

    try:
        message_content = response.output[0].content[0].text  # type: ignore
    except Exception:
        raise ValueError("response does not have a message content - response.output[0].content[0].text is not present")

    try:
        message_role = response.output[0].role  # type: ignore
    except Exception:
        raise ValueError("response does not have a message role - response.output[0].role is not present")

    try:
        logprobs_list = response.output[0].content[0].logprobs or []  # type: ignore
    except Exception:
        logprobs_list = []

    return ChatCompletion(
        id=response.id,
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content=message_content,
                    role=message_role,
                ),
                logprobs=ChoiceLogprobs(
                    content=[
                        ChatCompletionTokenLogprob(
                            token=lp.token,
                            bytes=lp.bytes,
                            logprob=lp.logprob,
                            top_logprobs=[
                                TopLogprob(
                                    token=top.token,
                                    bytes=top.bytes,
                                    logprob=top.logprob,
                                )
                                for top in lp.top_logprobs
                            ],
                        )
                        for lp in logprobs_list
                    ],
                    refusal=None,
                ),
            )
        ],
        created=int(response.created_at),
        model=response.model,
        object="chat.completion",
        usage=CompletionUsage(
            completion_tokens=response.usage.output_tokens,  # type: ignore
            prompt_tokens=response.usage.input_tokens,  # type: ignore
            total_tokens=response.usage.total_tokens,  # type: ignore
        ),
    )


def _responses_kwargs_to_chat_completion_kwargs(
    responses_kwargs: dict[str, Any],
) -> dict[str, Any]:
    try:
        from openai.lib._parsing._completions import type_to_response_format_param
    except ImportError as e:
        raise ImportError(
            "OpenAI is required to use the TLMResponses class. Please install it with `pip install openai`."
        ) from e

    chat_completion_kwargs = {}

    # TODO: also add "instruction" to the messages (system prompt)?
    if "input" in responses_kwargs:
        input_message = responses_kwargs["input"]

        if isinstance(input_message, str):
            chat_completion_kwargs["messages"] = [{"role": "user", "content": input_message}]

        elif isinstance(input_message, list):
            for message in input_message:
                content_item = message.get("content")
                if not (isinstance(content_item, list) and content_item and isinstance(content_item[0], dict)):
                    continue

                for item in content_item:
                    if isinstance(item, dict) and item.get("type") == "input_text":
                        item["type"] = "text"
                    elif isinstance(item, dict) and item.get("type") == "input_image":
                        item["type"] = "image_url"

            chat_completion_kwargs["messages"] = input_message

        else:
            chat_completion_kwargs["messages"] = input_message

    if "max_output_tokens" in responses_kwargs:
        chat_completion_kwargs["max_tokens"] = responses_kwargs["max_output_tokens"]

    if "reasoning" in responses_kwargs:
        reasoning_effort = responses_kwargs["reasoning"].get("effort")
        if reasoning_effort:
            chat_completion_kwargs["reasoning_effort"] = reasoning_effort

    if "text" in responses_kwargs:
        # for structured output
        chat_completion_kwargs["response_format"] = responses_kwargs["text"]["format"]

    if "text_format" in responses_kwargs:
        # for structured output
        chat_completion_kwargs["response_format"] = type_to_response_format_param(responses_kwargs["text_format"])  # type: ignore

    # TODO: handle tool calls properly, there is a lot of formatting logic here
    # if "tools" in responses_kwargs:
    #     chat_completion_kwargs["tools"] = responses_kwargs["tools"]

    # TODO: handle tool calls properly, there is a lot of formatting logic here
    # if "tool_choice" in responses_kwargs:
    #     chat_completion_kwargs["tool_choice"] = responses_kwargs["tool_choice"]

    shared_kwargs = {
        "metadata",
        "model",
        "parallel_tool_calls",
        "service_tier",
        "store",
        "stream",
        "temperature",
        "top_logprobs",
        "top_p",
        "user",
        "extra_headers",
        "extra_query",
        "extra_body",
        "timeout",
    }

    for kwarg in shared_kwargs:
        if kwarg in responses_kwargs:
            chat_completion_kwargs[kwarg] = responses_kwargs[kwarg]

    # ignored args: background, include, (instructions), max_tool_calls, previous_response_id, prompt, stream, truncation

    return chat_completion_kwargs

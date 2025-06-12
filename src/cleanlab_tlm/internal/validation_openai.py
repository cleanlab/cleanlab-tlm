from typing import Any

from cleanlab_tlm.internal.constants import _VALID_OPENAI_MODELS

VALID_KEYS = [
    "messages",
    "model",
    "temperature",
    "max_tokens",
    "logprobs",
    "top_logprobs",
    "top_p",
    "response_format",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "logit_bias",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "stop",
]


def validate_openai_kwargs(**openai_kwargs: Any) -> None:
    # check for unsupported keys
    invalid_keys = set(openai_kwargs.keys()) - set(VALID_KEYS)
    if invalid_keys:
        # TODO: one option is to just filter out the unsupported keys and return a warining + response with the unsupported keys removed
        raise ValueError(
            f"TLM currently does not support the following OpenAI parameters: {invalid_keys}. "
            f"Supported arguments are: {VALID_KEYS}"
        )

    if "messages" not in openai_kwargs:
        raise ValueError("messages parameter is required")
    if "model" not in openai_kwargs:
        raise ValueError("model parameter is required")

    # validate messages
    if not isinstance(openai_kwargs["messages"], (list, tuple)):
        raise TypeError("messages must be a sequence")
    if not all(isinstance(message, dict) for message in openai_kwargs["messages"]):
        raise TypeError("all messages must be dictionaries")

    # validate model
    if openai_kwargs["model"] not in _VALID_OPENAI_MODELS:
        raise ValueError(f"model must be one of {_VALID_OPENAI_MODELS}")

    # validate temperature
    if "temperature" in openai_kwargs:
        temp = openai_kwargs["temperature"]
        if not (
            temp is None
            or (isinstance(temp, float) and 0 <= temp <= 2)  # noqa: PLR2004
        ):
            raise TypeError("temperature must be None or a float between 0 and 2")

    # validate max_tokens
    if "max_tokens" in openai_kwargs:
        max_tokens_limit = 4096
        tokens = openai_kwargs["max_tokens"]
        if not (
            tokens is None
            or (isinstance(tokens, int) and 0 < tokens <= max_tokens_limit)
        ):
            raise TypeError(
                f"max_tokens must be None or a positive integer <= {max_tokens_limit}"
            )

    # validate logprobs
    if "logprobs" in openai_kwargs:
        if not (
            openai_kwargs["logprobs"] is None
            or isinstance(openai_kwargs["logprobs"], bool)
        ):
            raise TypeError("logprobs must be None or a boolean")

    # validate top_logprobs
    if "top_logprobs" in openai_kwargs:
        top_log = openai_kwargs["top_logprobs"]
        if not (
            top_log is None
            or (isinstance(top_log, int) and 0 < top_log <= 20)  # noqa: PLR2004
        ):
            raise TypeError("top_logprobs must be None or an integer between 1 and 20")

    # validate top_p
    if "top_p" in openai_kwargs:
        if not (
            openai_kwargs["top_p"] is None or isinstance(openai_kwargs["top_p"], float)
        ):
            raise TypeError("top_p must be None or a float")

    # validate response_format
    if "response_format" in openai_kwargs:
        if not (
            openai_kwargs["response_format"] is None
            or isinstance(openai_kwargs["response_format"], dict)
        ):
            raise TypeError("response_format must be None or a dictionary")

    # validate tools
    if "tools" in openai_kwargs:
        tools = openai_kwargs["tools"]
        if not (
            tools is None
            or (
                isinstance(tools, list)
                and all(isinstance(tool, dict) for tool in tools)
            )
        ):
            raise TypeError("tools must be None or a list of dictionaries")

    # validate tool_choice
    if "tool_choice" in openai_kwargs:
        if not (
            openai_kwargs["tool_choice"] is None
            or isinstance(openai_kwargs["tool_choice"], (str, dict))
        ):
            raise TypeError("tool_choice must be None, a string, or a dictionary")

    # validate parallel_tool_calls
    if "parallel_tool_calls" in openai_kwargs:
        if not (
            openai_kwargs["parallel_tool_calls"] is None
            or isinstance(openai_kwargs["parallel_tool_calls"], bool)
        ):
            raise TypeError("parallel_tool_calls must be None or a boolean")

    # validate logit_bias
    if "logit_bias" in openai_kwargs:
        if not (
            openai_kwargs["logit_bias"] is None
            or isinstance(openai_kwargs["logit_bias"], dict)
        ):
            raise TypeError("logit_bias must be None or a dictionary")

    # validate frequency_penalty
    if "frequency_penalty" in openai_kwargs:
        freq = openai_kwargs["frequency_penalty"]
        if not (
            freq is None
            or (isinstance(freq, float) and -2 <= freq <= 2)  # noqa: PLR2004
        ):
            raise TypeError(
                "frequency_penalty must be None or a float between -2 and 2"
            )

    # validate presence_penalty
    if "presence_penalty" in openai_kwargs:
        pres = openai_kwargs["presence_penalty"]
        if not (
            pres is None
            or (isinstance(pres, float) and -2 <= pres <= 2)  # noqa: PLR2004
        ):
            raise TypeError("presence_penalty must be None or a float between -2 and 2")

    # validate seed
    if "seed" in openai_kwargs:
        if not (
            openai_kwargs["seed"] is None or isinstance(openai_kwargs["seed"], int)
        ):
            raise TypeError("seed must be None or an integer")

    # validate stop
    if "stop" in openai_kwargs:
        if not (
            openai_kwargs["stop"] is None
            or isinstance(openai_kwargs["stop"], (str, list))
        ):
            raise TypeError("stop must be None, a string, or a list")

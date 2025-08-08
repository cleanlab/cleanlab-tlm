"""Compatibility layer for OpenAI SDK tool call types across versions.

This module exposes two names for tests to import:
- ChatCompletionMessageToolCall: the class representing a function tool call
    In previous versions, this was called `ChatCompletionMessageToolCall`,
    but that has now become a union for both Custom Tool Calls and Function Tool Calls,
    which have different schemas.
    This is a shim to allow tests to work with both types.
- Function: the inner function payload model
    The Function model is only used for Function Tool Calls, which are currently supported
    by this package.

Works with SDKs that expose either the legacy
`chat_completion_message_tool_call.ChatCompletionMessageToolCall` or the
newer `chat_completion_message_function_tool_call.ChatCompletionMessageFunctionToolCall`.
"""

from __future__ import annotations

try:  # OpenAI SDK >= 1.99.2
    from openai.types.chat.chat_completion_message_function_tool_call import (
        ChatCompletionMessageFunctionToolCall as ChatCompletionMessageToolCall,
    )
    from openai.types.chat.chat_completion_message_function_tool_call import (
        Function,
    )
except Exception:  # OpenAI SDK <= 1.99.1
    import importlib

    _legacy = importlib.import_module("openai.types.chat.chat_completion_message_tool_call")
    ChatCompletionMessageToolCall = _legacy.ChatCompletionMessageToolCall  # type: ignore
    Function = _legacy.Function  # type: ignore

__all__ = ["ChatCompletionMessageToolCall", "Function"]

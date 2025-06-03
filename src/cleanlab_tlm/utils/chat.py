"""Utilities for formatting chat messages into prompt strings.

This module provides helper functions for working with chat messages in the format used by
OpenAI's chat models.
"""

import json
import warnings
from typing import Any, Optional


def _format_tools_prompt(tools: list[dict[str, Any]], is_responses: bool = False) -> str:
    """
    Format a list of tool definitions into a system message with tools.

    Args:
        tools (List[Dict[str, Any]]): The list of tools made available for the LLM to use when responding to the messages.
            This is the same argument as the tools argument for OpenAI's Responses API or Chat Completions API.
            This list of tool definitions will be formatted into a system message.
        is_responses (bool): Whether the tools are in responses API format.

    Returns:
        str: Formatted string with tools as a system message.
    """
    system_message = (
        "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. "
        "You may call one or more functions to assist with the user query. If available tools are not relevant in assisting "
        "with user query, just respond in natural conversational language. Don't make assumptions about what values to plug "
        "into functions. After calling & executing the functions, you will be provided with function results within "
        "<tool_response> </tool_response> XML tags.\n\n"
        "<tools>\n"
    )

    # Format each tool as a function spec
    tool_strings = []
    for tool in tools:
        if not is_responses:
            tool_dict = {
                "type": "function",
                "function": {
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "parameters": tool["function"]["parameters"],
                },
            }
        else:  # responses format
            tool_dict = {
                "type": "function",
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
                "strict": tool.get("strict", True),
            }
        tool_strings.append(json.dumps(tool_dict, separators=(",", ":")))

    system_message += "\n".join(tool_strings)
    system_message += "\n</tools>\n\n"

    # Add function call schema and example
    system_message += (
        "For each function call return a JSON object, with the following pydantic model json schema:\n"
        "{'name': <function-name>, 'arguments': <args-dict>}\n"
        "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
        "Example:\n"
        "<tool_call>\n"
        "{'name': <function-name>, 'arguments': <args-dict>}\n"
        "</tool_call>\n\n"
        "Note: Your past messages will include a call_id in the <tool_call> XML tags. "
        "However, do not generate your own call_id when making a function call."
    )

    return f"System: {system_message}"


def _uses_responses_api(messages: list[dict[str, Any]], tools: Optional[list[dict[str, Any]]] = None) -> bool:
    """
    Determine if the messages and tools are in responses API format.

    Args:
        messages (List[Dict]): A list of dictionaries representing chat messages.
        tools (Optional[List[Dict[str, Any]]]): The list of tools made available for the LLM.

    Returns:
        bool: True if using responses API format, False if using chat completions API format.
    """
    return any(msg.get("type") in ["function_call", "function_call_output"] for msg in messages) or any(
        "name" in tool and "function" not in tool for tool in tools or []
    )


def _form_prompt_responses_api(messages: list[dict[str, Any]], tools: Optional[list[dict[str, Any]]] = None) -> str:
    """
    Convert messages in [OpenAI Responses API format](https://platform.openai.com/docs/api-reference/responses) into a single prompt string.

    Args:
        messages (List[Dict]): A list of dictionaries representing chat messages in responses API format.
        tools (Optional[List[Dict[str, Any]]]): The list of tools made available for the LLM to use when responding to the messages.
        This is the same argument as the tools argument for OpenAI's Responses API.
        This list of tool definitions will be formatted into a system message.

    Returns:
        str: A formatted string representing the chat history as a single prompt.
    """
    output = ""
    if tools is not None:
        output = _format_tools_prompt(tools, is_responses=True) + "\n\n"

    # Only return content directly if there's a single user message AND no tools
    if len(messages) == 1 and messages[0].get("role") == "user" and tools is None:
        return str(output + messages[0]["content"])

    # Warn if the last message is a tool call
    if messages and messages[-1].get("type") == "function_call":
        warnings.warn(
            "The last message is a tool call or assistant message. The next message should not be an LLM response. "
            "This prompt should not be used for trustworthiness scoring.",
            UserWarning,
            stacklevel=2,
        )

    # Track function names by call_id for function call outputs
    function_names = {}

    for msg in messages:
        if "type" in msg:
            if msg["type"] == "function_call":
                output += "Assistant: "
                # If there's content in the message, add it before the tool call
                if msg.get("content"):
                    output += f"{msg['content']}\n\n"
                call_id = msg.get("call_id", "")
                function_names[call_id] = msg["name"]
                # Format function call as JSON within XML tags, now including call_id
                function_call = {"name": msg["name"], "arguments": json.loads(msg["arguments"]), "call_id": call_id}
                output += f"<tool_call>\n{json.dumps(function_call, indent=2)}\n</tool_call>\n\n"
            elif msg["type"] == "function_call_output":
                call_id = msg.get("call_id", "")
                name = function_names.get(call_id, "function")
                # Format function response as JSON within XML tags
                tool_response = {"name": name, "call_id": call_id, "output": msg["output"]}
                output += f"<tool_response>\n{json.dumps(tool_response, indent=2)}\n</tool_response>\n\n"
        else:
            role = msg.get("name", msg["role"])
            if role == "system":
                prefix = "System: "
            elif role == "user":
                prefix = "User: "
            elif role == "assistant":
                prefix = "Assistant: "
            else:
                prefix = role.capitalize() + ": "
            output += f"{prefix}{msg['content']}\n\n"

    output += "Assistant:"
    return output.strip()


def _form_prompt_chat_completions_api(
    messages: list[dict[str, Any]], tools: Optional[list[dict[str, Any]]] = None
) -> str:
    """
    Convert messages in [OpenAI Chat Completions API format](https://platform.openai.com/docs/api-reference/chat) into a single prompt string.

    Args:
        messages (List[Dict]): A list of dictionaries representing chat messages in chat completions API format.
        tools (Optional[List[Dict[str, Any]]]): The list of tools made available for the LLM to use when responding to the messages.
        This is the same argument as the tools argument for OpenAI's Chat Completions API.
        This list of tool definitions will be formatted into a system message.

    Returns:
        str: A formatted string representing the chat history as a single prompt.
    """
    output = ""
    if tools is not None:
        output = _format_tools_prompt(tools, is_responses=False) + "\n\n"

    # Only return content directly if there's a single user message AND no tools
    if len(messages) == 1 and messages[0].get("role") == "user" and tools is None:
        return str(output + messages[0]["content"])

    # Warn if the last message is an assistant message with tool calls
    if messages and (messages[-1].get("role") == "assistant" or "tool_calls" in messages[-1]):
        warnings.warn(
            "The last message is a tool call or assistant message. The next message should not be an LLM response. "
            "This prompt should not be used for trustworthiness scoring.",
            UserWarning,
            stacklevel=2,
        )

    # Track function names by call_id for function call outputs
    function_names = {}

    for msg in messages:
        if msg["role"] == "assistant":
            output += "Assistant: "
            # Handle content if present
            if msg.get("content"):
                output += f"{msg['content']}\n\n"
            # Handle tool calls if present
            if "tool_calls" in msg:
                for tool_call in msg["tool_calls"]:
                    call_id = tool_call["id"]
                    function_names[call_id] = tool_call["function"]["name"]
                    # Format function call as JSON within XML tags, now including call_id
                    function_call = {
                        "name": tool_call["function"]["name"],
                        "arguments": json.loads(tool_call["function"]["arguments"]),
                        "call_id": call_id,
                    }
                    output += f"<tool_call>\n{json.dumps(function_call, indent=2)}\n</tool_call>\n\n"
        elif msg["role"] == "tool":
            # Handle tool responses
            call_id = msg["tool_call_id"]
            name = function_names.get(call_id, "function")
            # Format function response as JSON within XML tags
            tool_response = {"name": name, "call_id": call_id, "output": msg["content"]}
            output += f"<tool_response>\n{json.dumps(tool_response, indent=2)}\n</tool_response>\n\n"
        else:
            role = msg["role"]
            if role == "system":
                prefix = "System: "
            elif role == "user":
                prefix = "User: "
            else:
                prefix = role.capitalize() + ": "
            output += f"{prefix}{msg['content']}\n\n"

    output += "Assistant:"
    return output.strip()


def form_prompt_string(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    use_responses: Optional[bool] = None,
) -> str:
    """
    Convert a list of chat messages into a single string prompt.

    If there is only one message and no tools are provided, returns the content directly.
    Otherwise, concatenates all messages with appropriate role prefixes and ends with
    "Assistant:" to indicate the assistant's turn is next.

    If tools are provided, they will be formatted as a system message at the start
    of the prompt. In this case, even a single message will use role prefixes since
    there will be at least one system message (the tools section).

    Handles messages in either OpenAI's [Responses API](https://platform.openai.com/docs/api-reference/responses) or [Chat Completions API](https://platform.openai.com/docs/api-reference/chat) formats.

    Args:
        messages (List[Dict]): A list of dictionaries representing chat messages.
            Each dictionary should contain either:
            For responses API:
            - 'role' and 'content' for regular messages
            - 'type': 'function_call' and function call details for tool calls
            - 'type': 'function_call_output' and output details for tool results
            For chat completions API:
            - 'role': 'user', 'assistant', 'system', or 'tool' and appropriate content
            - For assistant messages with tool calls: 'tool_calls' containing function calls
            - For tool messages: 'tool_call_id' and 'content' for tool responses
        tools (Optional[List[Dict[str, Any]]]): The list of tools made available for the LLM to use when responding to the messages.
            This is the same argument as the tools argument for OpenAI's Responses API or Chat Completions API.
            This list of tool definitions will be formatted into a system message.
        use_responses (Optional[bool]): If provided, explicitly specifies whether to use responses API format.
            If None, the format is automatically detected using _uses_responses_api.

    Returns:
        str: A formatted string representing the chat history as a single prompt.
    """
    is_responses = use_responses if use_responses is not None else _uses_responses_api(messages, tools)
    return (
        _form_prompt_responses_api(messages, tools)
        if is_responses
        else _form_prompt_chat_completions_api(messages, tools)
    )

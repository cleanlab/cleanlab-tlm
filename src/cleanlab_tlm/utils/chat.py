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
        tools (List[Dict[str, Any]]): List of OpenAI tool/function specs.
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
        "{'name': <function-name>, 'arguments': <args-dict>, 'call_id': <call-id>}\n"
        "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
        "Example:\n"
        "<tool_call>\n"
        "{'name': <function-name>, 'arguments': <args-dict>, 'call_id': <call-id>}\n"
        "</tool_call>"
    )

    return f"System: {system_message}"


def form_prompt_string(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
) -> str:
    """
    Convert a list of chat messages into a single string prompt.

    If there is only one message and no tools are provided, returns the content directly.
    Otherwise, concatenates all messages with appropriate role prefixes and ends with
    "Assistant:" to indicate the assistant's turn is next.

    If tools are provided, they will be formatted as a system message at the start
    of the prompt. In this case, even a single message will use role prefixes since
    there will be at least one system message (the tools section).

    Handles both OpenAI's responses API and chat completions API formats:
    - responses API: Uses 'type' field with 'function_call' and 'function_call_output'
    - chat completions API: Uses 'role' field with 'assistant' and 'tool', and 'tool_calls' for function calls

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
        tools (Optional[List[Dict[str, Any]]]): Optional list of OpenAI tool/function specs to include
            as a system message at the start of the prompt.

    Returns:
        str: A formatted string representing the chat history as a single prompt.
    """
    # Check which format we're using by looking for responses format indicators in both messages and tools
    is_responses = any(
        msg.get("type") in ["function_call", "function_call_output"] for msg in messages
    ) or any(
        "name" in tool and "function" not in tool for tool in tools or []
    )

    output = ""
    if tools is not None:
        output = _format_tools_prompt(tools, is_responses) + "\n\n"

    # Only return content directly if there's a single user message AND no tools
    if len(messages) == 1 and messages[0].get("role") == "user" and tools is None:
        return str(output + messages[0]["content"])

    # Warn if the last message is a tool call or assistant message
    if messages:
        last_msg = messages[-1]
        if is_responses:
            if "type" in last_msg and last_msg["type"] == "function_call":
                warnings.warn(
                    "The last message is a tool call or assistant message. The next message should not be an LLM response. "
                    "This prompt should not be used for trustworthiness scoring.",
                    UserWarning,
                    stacklevel=2,
                )
        else:  # chat completions
            if last_msg.get("role") == "assistant" or "tool_calls" in last_msg:
                warnings.warn(
                    "The last message is a tool call or assistant message. The next message should not be an LLM response. "
                    "This prompt should not be used for trustworthiness scoring.",
                    UserWarning,
                    stacklevel=2,
                )

    # Track function names by call_id for function call outputs
    function_names = {}

    for msg in messages:
        # Handle tool calls and function call outputs differently for responses and chat completions
        if is_responses:
            if "type" in msg:
                if msg["type"] == "function_call":
                    call_id = msg.get("call_id", "")
                    function_names[call_id] = msg["name"]
                    # Format function call as JSON within XML tags, now including call_id
                    function_call = {
                        "name": msg["name"],
                        "arguments": json.loads(msg["arguments"]),
                        "call_id": call_id
                    }
                    output += f"Assistant: <tool_call>\n{json.dumps(function_call, indent=2)}\n</tool_call>\n\n"
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
        else:  # chat_completions format
            if msg["role"] == "assistant":
                # Handle tool calls if present
                if "tool_calls" in msg:
                    for tool_call in msg["tool_calls"]:
                        call_id = tool_call["id"]
                        function_names[call_id] = tool_call["function"]["name"]
                        # Format function call as JSON within XML tags, now including call_id
                        function_call = {
                            "name": tool_call["function"]["name"],
                            "arguments": json.loads(tool_call["function"]["arguments"]),
                            "call_id": call_id
                        }
                        output += f"Assistant: <tool_call>\n{json.dumps(function_call, indent=2)}\n</tool_call>\n\n"
                # Handle content if present
                if msg.get("content"):
                    output += f"Assistant: {msg['content']}\n\n"
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

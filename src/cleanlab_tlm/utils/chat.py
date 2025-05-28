"""Utilities for formatting chat messages into prompt strings.

This module provides helper functions for working with chat messages in the format used by
OpenAI's chat models.
"""

import json
import warnings
from typing import Any, Optional


def format_tools_prompt(tools: list[dict[str, Any]]) -> str:
    """
    Format a list of tool definitions into a system message with tools.

    Args:
        tools (List[Dict[str, Any]]): List of OpenAI tool/function specs.

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
        tool_dict = {
            "type": "function",
            "function": {"name": tool["name"], "description": tool["description"], "parameters": tool["parameters"]},
        }
        tool_strings.append(json.dumps(tool_dict, separators=(",", ":")))

    system_message += "\n".join(tool_strings)
    system_message += "\n</tools>\n\n"

    # Add function call schema and example
    system_message += (
        "For each function call return a JSON object, with the following pydantic model json schema:\n"
        "{'title': 'FunctionCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, "
        "'arguments': {'title': 'Arguments', 'type': 'object'}}, 'required': ['arguments', 'name']}\n"
        "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
        "Example:\n"
        "<tool_call>\n"
        "{'name': <function-name>, 'arguments': <args-dict>}\n"
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

    Handles special message types:
    - function_call: Tool/function calls made by the assistant
    - function_call_output: Outputs/results from tool/function calls

    Args:
        messages (List[Dict]): A list of dictionaries representing chat messages.
            Each dictionary should contain either:
            - 'role' and 'content' for regular messages
            - 'type': 'function_call' and function call details for tool calls
            - 'type': 'function_call_output' and output details for tool results
        tools (Optional[List[Dict[str, Any]]]): Optional list of OpenAI tool/function specs to include
            as a system message at the start of the prompt.

    Returns:
        str: A formatted string representing the chat history as a single prompt.

    Example:
        ```python
        messages = [
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "type": "function_call",
                "name": "get_weather",
                "call_id": "call_12345xyz",
                "arguments": '{"latitude":48.8566,"longitude":2.3522}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_12345xyz",
                "output": "22.1",
            },
        ]
        tools = [{"name": "get_weather", "description": "Get weather for location"}]
        result = form_prompt_string(messages, tools)
        print(result)
        # System: You are a function calling AI model...
        # <tools>
        # {"type":"function","function":{"name":"get_weather","description":"Get weather for location"}}
        # </tools>
        #
        # User: What's the weather in Paris?
        #
        # Assistant: <tool_call>
        # {"name": "get_weather", "arguments": {"latitude": 48.8566, "longitude": 2.3522}}
        # </tool_call>
        #
        # <tool_response>
        # {"name": "get_weather", "call_id": "call_12345xyz", "output": "22.1"}
        # </tool_response>
        #
        # Assistant:
        ```
    """
    output = ""
    if tools is not None:
        output = format_tools_prompt(tools) + "\n\n"

    # Only return content directly if there's a single user message AND no tools
    if len(messages) == 1 and messages[0]["role"] == "user" and tools is None:
        return str(output + messages[0]["content"])

    # Warn if the last message is a tool call or assistant message
    if messages:
        last_msg = messages[-1]
        if ("type" in last_msg and last_msg["type"] == "function_call") or (last_msg.get("role") == "assistant"):
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
            # Handle function calls and outputs
            if msg["type"] == "function_call":
                call_id = msg.get("call_id", "")
                function_names[call_id] = msg["name"]
                # Format function call as JSON within XML tags
                function_call = {"name": msg["name"], "arguments": json.loads(msg["arguments"])}
                output += f"Assistant: <tool_call>\n{json.dumps(function_call, indent=2)}\n</tool_call>\n\n"
            elif msg["type"] == "function_call_output":
                call_id = msg.get("call_id", "")
                name = function_names.get(call_id, "function")
                # Format function response as JSON within XML tags
                tool_response = {"name": name, "call_id": call_id, "output": msg["output"]}
                output += f"<tool_response>\n{json.dumps(tool_response, indent=2)}\n</tool_response>\n\n"
        else:
            # Handle regular messages
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

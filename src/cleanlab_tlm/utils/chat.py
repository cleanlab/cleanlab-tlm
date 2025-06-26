"""Utilities for formatting chat messages into prompt strings.

This module provides helper functions for working with chat messages in the format used by
OpenAI's chat models.
"""

import json
import warnings
from typing import TYPE_CHECKING, Any, Literal, Optional, cast

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

# Define message prefixes
_SYSTEM_PREFIX = "System: "
_USER_PREFIX = "User: "
_ASSISTANT_PREFIX = "Assistant: "
_TOOL_PREFIX = "Tool: "

# Define role constants
_SYSTEM_ROLE: Literal["system"] = "system"
_DEVELOPER_ROLE: Literal["developer"] = "developer"
_USER_ROLE: Literal["user"] = "user"
_TOOL_ROLE: Literal["tool"] = "tool"
_ASSISTANT_ROLE: Literal["assistant"] = "assistant"


# Define system roles
_SYSTEM_ROLES = [_SYSTEM_ROLE, _DEVELOPER_ROLE]

# Define message type constants
_FUNCTION_CALL_TYPE = "function_call"
_FUNCTION_CALL_OUTPUT_TYPE = "function_call_output"

# Define XML tag constants
_TOOLS_TAG_START = "<tools>"
_TOOLS_TAG_END = "</tools>"
_TOOL_CALL_TAG_START = "<tool_call>"
_TOOL_CALL_TAG_END = "</tool_call>"
_TOOL_RESPONSE_TAG_START = "<tool_response>"
_TOOL_RESPONSE_TAG_END = "</tool_response>"

# Define tool-related message prefixes
_TOOL_DEFINITIONS_PREFIX = (
    "You are an AI Assistant that can call provided tools (a.k.a. functions). "
    "The set of available tools is provided to you as function signatures within "
    f"{_TOOLS_TAG_START} {_TOOLS_TAG_END} XML tags. "
    "You may call one or more of these functions to assist with the user query. If the provided functions are not helpful/relevant, "
    "then just respond in natural conversational language. Don't make assumptions about what values to plug "
    "into functions. After you choose to call a function, you will be provided with the function's results within "
    f"{_TOOL_RESPONSE_TAG_START} {_TOOL_RESPONSE_TAG_END} XML tags.\n\n"
    f"{_TOOLS_TAG_START}\n"
)

_TOOL_CALL_SCHEMA_PREFIX = (
    "For each function call return a JSON object, with the following pydantic model json schema:\n"
    "{'name': <function-name>, 'arguments': <args-dict>}\n"
    f"Each function call should be enclosed within {_TOOL_CALL_TAG_START} {_TOOL_CALL_TAG_END} XML tags.\n"
    "Example:\n"
    f"{_TOOL_CALL_TAG_START}\n"
    "{'name': <function-name>, 'arguments': <args-dict>}\n"
    f"{_TOOL_CALL_TAG_END}\n\n"
    "Note: Your past messages will include a call_id in the "
    f"{_TOOL_CALL_TAG_START} XML tags. "
    "However, do not generate your own call_id when making a function call."
)


def _format_tools_prompt(tools: list[dict[str, Any]], is_responses: bool = False) -> str:
    """
    Format a list of tool definitions into a system message with tools.

    Args:
        tools (List[Dict[str, Any]]): The list of tools made available for the LLM to use when responding to the messages.
            This is the same argument as the tools argument for OpenAI's Responses API or Chat Completions API.
            This list of tool definitions will be formatted into a system message.
        is_responses (bool): Whether the tools are in Responses API format.

    Returns:
        str: Formatted string with tools as a system message.
    """
    system_message = _TOOL_DEFINITIONS_PREFIX

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
    system_message += f"\n{_TOOLS_TAG_END}\n\n"
    system_message += _TOOL_CALL_SCHEMA_PREFIX

    return system_message


def _uses_responses_api(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    use_responses: Optional[bool] = None,
    **responses_api_kwargs: Any,
) -> bool:
    """
    Determine if the messages and parameters indicate Responses API format.

    Args:
        messages (List[Dict]): A list of dictionaries representing chat messages.
        tools (Optional[List[Dict[str, Any]]]): The list of tools made available for the LLM.
        use_responses (Optional[bool]): If provided, explicitly specifies whether to use Responses API format.
            Cannot be set to False when Responses API kwargs are provided.
        **responses_api_kwargs: Optional keyword arguments for OpenAI's Responses API. Currently supported:
            - instructions (str): Developer instructions to prepend to the prompt with highest priority.

    Returns:
        bool: True if using Responses API format, False if using chat completions API format.

    Raises:
        ValueError: If Responses API kwargs are provided with use_responses=False.
    """
    # First check if explicitly set to False while having Responses API kwargs
    if use_responses is False and responses_api_kwargs:
        raise ValueError(
            "Responses API kwargs are only supported in Responses API format. Cannot use with use_responses=False."
        )

    # If explicitly set to True or False, respect that (after validation above)
    if use_responses is not None:
        return use_responses

    # Check for Responses API kwargs
    responses_api_keywords = {"instructions"}
    if any(key in responses_api_kwargs for key in responses_api_keywords):
        return True

    # Check messages for Responses API format indicators
    if any(msg.get("type") in [_FUNCTION_CALL_TYPE, _FUNCTION_CALL_OUTPUT_TYPE] for msg in messages):
        return True

    # Check tools for Responses API format indicators
    if tools and any("name" in tool and "function" not in tool for tool in tools):
        return True

    return False


def _get_prefix(msg: dict[str, Any], prev_msg_role: Optional[str] = None) -> str:
    """
    Get the appropriate prefix for a message based on its role.

    Args:
        msg (Dict[str, Any]): A message dictionary containing at least a 'role' key.
        prev_msg_role (Optional[str]): The role of the previous message, if any.

    Returns:
        str: The appropriate prefix for the message role.
    """
    role = str(msg.get("name", msg["role"]))

    # Skip prefix for system messages if the previous message was also a system message
    if role in _SYSTEM_ROLES and prev_msg_role in _SYSTEM_ROLES:
        return ""

    if role in _SYSTEM_ROLES:
        return _SYSTEM_PREFIX
    if role == _USER_ROLE:
        return _USER_PREFIX
    if role == _ASSISTANT_ROLE:
        return _ASSISTANT_PREFIX
    return role.capitalize() + ": "


def _find_index_after_first_system_block(messages: list[dict[str, Any]]) -> int:
    """
    Find the index after the first consecutive block of system messages.

    Args:
        messages (List[Dict]): A list of message dictionaries.

    Returns:
        int: The index after the first consecutive block of system messages.
             Returns -1 if no system messages are found.
    """
    last_system_idx = -1
    for i, msg in enumerate(messages):
        if msg.get("role") in _SYSTEM_ROLES:
            last_system_idx = i
        else:
            # Found a non-system message, so we've reached the end of the first system block
            break

    return last_system_idx


def _form_prompt_responses_api(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    **responses_api_kwargs: Any,
) -> str:
    """
    Convert messages in [OpenAI Responses API format](https://platform.openai.com/docs/api-reference/responses) into a single prompt string.

    Args:
        messages (List[Dict]): A list of dictionaries representing chat messages in Responses API format.
        tools (Optional[List[Dict[str, Any]]]): The list of tools made available for the LLM to use when responding to the messages.
            This is the same argument as the tools argument for OpenAI's Responses API.
            This list of tool definitions will be formatted into a system message.
        **responses_api_kwargs: Optional keyword arguments for OpenAI's Responses API. Currently supported:
            - instructions (str): Developer instructions to prepend to the prompt with highest priority.

    Returns:
        str: A formatted string representing the chat history as a single prompt.
    """
    messages = messages.copy()
    output = ""

    # Find the index after the first consecutive block of system messages
    last_system_idx = _find_index_after_first_system_block(messages)

    # Insert tool definitions and instructions after system messages if needed
    if tools is not None and len(tools) > 0:
        messages.insert(
            last_system_idx + 1,
            {
                "role": _SYSTEM_ROLE,
                "content": _format_tools_prompt(tools, is_responses=True),
            },
        )

    if "instructions" in responses_api_kwargs:
        messages.insert(0, {"role": _SYSTEM_ROLE, "content": responses_api_kwargs["instructions"]})

    # Only return content directly if there's a single user message AND no prepended content
    if len(messages) == 1 and messages[0].get("role") == _USER_ROLE and not output:
        return str(messages[0]["content"])

    # Warn if the last message is a tool call
    if messages and messages[-1].get("type") == _FUNCTION_CALL_TYPE:
        warnings.warn(
            "The last message is a tool call or assistant message. The next message should not be an LLM response. "
            "This prompt should not be used for trustworthiness scoring.",
            UserWarning,
            stacklevel=2,
        )

    # Track function names by call_id for function call outputs
    function_names = {}
    prev_msg_role = None

    for msg in messages:
        if "type" in msg:
            if msg["type"] == _FUNCTION_CALL_TYPE:
                output += _ASSISTANT_PREFIX
                # If there's content in the message, add it before the tool call
                if msg.get("content"):
                    output += f"{msg['content']}\n\n"
                call_id = msg.get("call_id", "")
                function_names[call_id] = msg["name"]
                # Format function call as JSON within XML tags, now including call_id
                function_call = {
                    "name": msg["name"],
                    "arguments": json.loads(msg["arguments"]) if msg["arguments"] else {},
                    "call_id": call_id,
                }
                output += f"{_TOOL_CALL_TAG_START}\n{json.dumps(function_call, indent=2)}\n{_TOOL_CALL_TAG_END}\n\n"
            elif msg["type"] == _FUNCTION_CALL_OUTPUT_TYPE:
                output += _TOOL_PREFIX
                call_id = msg.get("call_id", "")
                name = function_names.get(call_id, "function")
                # Format function response as JSON within XML tags
                tool_response = {
                    "name": name,
                    "call_id": call_id,
                    "output": msg["output"],
                }
                output += (
                    f"{_TOOL_RESPONSE_TAG_START}\n{json.dumps(tool_response, indent=2)}\n{_TOOL_RESPONSE_TAG_END}\n\n"
                )
        else:
            prefix = _get_prefix(msg, prev_msg_role)
            output += f"{prefix}{msg['content']}\n\n"
            prev_msg_role = msg["role"]

    output += _ASSISTANT_PREFIX
    return output.strip()


def _form_prompt_chat_completions_api(
    messages: list["ChatCompletionMessageParam"],
    tools: Optional[list[dict[str, Any]]] = None,
) -> str:
    """
    Convert messages in [OpenAI Chat Completions API format](https://platform.openai.com/docs/api-reference/chat) into a single prompt string.

    Args:
        messages (List[ChatCompletionsMessageParam]): A list of dictionaries representing chat messages in chat completions API format.
        tools (Optional[List[Dict[str, Any]]]): The list of tools made available for the LLM to use when responding to the messages.
        This is the same argument as the tools argument for OpenAI's Chat Completions API.
        This list of tool definitions will be formatted into a system message.

    Returns:
        str: A formatted string representing the chat history as a single prompt.
    """
    messages = messages.copy()
    output = ""

    # Find the index after the first consecutive block of system messages
    last_system_idx = _find_index_after_first_system_block(cast(list[dict[str, Any]], messages))

    if tools is not None and len(tools) > 0:
        messages.insert(
            last_system_idx + 1,
            {
                "role": "system",
                "content": _format_tools_prompt(tools, is_responses=False),
            },
        )

    # Only return content directly if there's a single user message AND no tools
    if len(messages) == 1 and messages[0].get("role") == _USER_ROLE and (tools is None or len(tools) == 0):
        return output + str(messages[0]["content"])

    # Warn if the last message is an assistant message with tool calls
    if messages and (messages[-1].get("role") == _ASSISTANT_ROLE or "tool_calls" in messages[-1]):
        warnings.warn(
            "The last message is a tool call or assistant message. The next message should not be an LLM response. "
            "This prompt should not be used for trustworthiness scoring.",
            UserWarning,
            stacklevel=2,
        )

    # Track function names by call_id for function call outputs
    function_names = {}
    prev_msg_role = None

    for msg in messages:
        if msg["role"] == _ASSISTANT_ROLE:
            output += _ASSISTANT_PREFIX
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
                        "arguments": json.loads(tool_call["function"]["arguments"])
                        if tool_call["function"]["arguments"]
                        else {},
                        "call_id": call_id,
                    }
                    output += f"{_TOOL_CALL_TAG_START}\n{json.dumps(function_call, indent=2)}\n{_TOOL_CALL_TAG_END}\n\n"
        elif msg["role"] == _TOOL_ROLE:
            # Handle tool responses
            output += _TOOL_PREFIX
            call_id = msg["tool_call_id"]
            name = function_names.get(call_id, "function")
            # Format function response as JSON within XML tags
            tool_response = {"name": name, "call_id": call_id, "output": msg["content"]}
            output += f"{_TOOL_RESPONSE_TAG_START}\n{json.dumps(tool_response, indent=2)}\n{_TOOL_RESPONSE_TAG_END}\n\n"
        else:
            prefix = _get_prefix(cast(dict[str, Any], msg), prev_msg_role)
            output += f"{prefix}{msg['content']}\n\n"
            prev_msg_role = msg["role"]

    output += _ASSISTANT_PREFIX
    return output.strip()


def form_prompt_string(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    use_responses: Optional[bool] = None,
    **responses_api_kwargs: Any,
) -> str:
    """
    Convert a list of chat messages into a single string prompt.

    If there is only one message and no tools are provided, returns the content directly.
    Otherwise, concatenates all messages with appropriate role prefixes and ends with
    "Assistant:" to indicate the assistant's turn is next.

    If tools are provided, they will be formatted as a system message at the start
    of the prompt. In this case, even a single message will use role prefixes since
    there will be at least one system message (the tools section).

    If Responses API kwargs (like instructions) are provided, they will be
    formatted for the Responses API format. These kwargs are only supported
    for the Responses API format.

    Handles messages in either OpenAI's [Responses API](https://platform.openai.com/docs/api-reference/responses) or [Chat Completions API](https://platform.openai.com/docs/api-reference/chat) formats.

    Args:
        messages (List[Dict]): A list of dictionaries representing chat messages.
            Each dictionary should contain either:
            For Responses API:
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
        use_responses (Optional[bool]): If provided, explicitly specifies whether to use Responses API format.
            If None, the format is automatically detected using _uses_responses_api.
            Cannot be set to False when Responses API kwargs are provided.
        **responses_api_kwargs: Optional keyword arguments for OpenAI's Responses API. Currently supported:
            - instructions (str): Developer instructions to prepend to the prompt with highest priority.

    Returns:
        str: A formatted string representing the chat history as a single prompt.

    Raises:
        ValueError: If Responses API kwargs are provided with use_responses=False.
    """
    is_responses = _uses_responses_api(messages, tools, use_responses, **responses_api_kwargs)

    return (
        _form_prompt_responses_api(messages, tools, **responses_api_kwargs)
        if is_responses
        else _form_prompt_chat_completions_api(cast(list["ChatCompletionMessageParam"], messages), tools)
    )


def form_response_string_chat_completions_api(response: dict[str, Any]) -> str:
    """
    Format an assistant response message dictionary from the Chat Completions API into a single string.

    This function takes a response.choices[0].message.to_dict() from a chat.completions.create()
    and formats it into a string that includes both content and tool calls (if present).
    Tool calls are formatted using XML tags with JSON content, consistent with the format
    used in `form_prompt_string`.

    Args:
        response (dict[str, Any]): A chat completion response message dictionary, containing:
            - 'content' (str): The main response content from the LLM
            - 'tool_calls' (List[Dict], optional): List of tool calls made by the LLM,
              where each tool call contains function name and arguments

    Returns:
        str: A formatted string containing the response content and any tool calls.
             Tool calls are formatted as XML tags containing JSON with function
             name and arguments.

    Raises:
        TypeError: If response is not a dictionary.
    """
    if not isinstance(response, dict):
        raise TypeError(f"Expected response to be a dict, got {type(response).__name__}")

    content = response.get("content") or ""

    if "tool_calls" in response:
        try:
            tool_calls = "\n".join(
                f"{_TOOL_CALL_TAG_START}\n{json.dumps({'name': call['function']['name'], 'arguments': json.loads(call['function']['arguments']) if call['function']['arguments'] else {}}, indent=2)}\n{_TOOL_CALL_TAG_END}"
                for call in response["tool_calls"]
            )
            return f"{content}\n{tool_calls}".strip() if content else tool_calls
        except (KeyError, TypeError, json.JSONDecodeError) as e:
            # Log the error but continue with just the content
            warnings.warn(
                f"Error formatting tool_calls in response: {e}. Returning content only.",
                UserWarning,
                stacklevel=2,
            )

    return str(content)

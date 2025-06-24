from typing import TYPE_CHECKING, Any, cast

import pytest

from cleanlab_tlm.utils.chat import (
    _form_prompt_chat_completions_api,
    _form_prompt_responses_api,
    form_prompt_string,
    form_response_string_chat_completions_api,
)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam


def test_form_prompt_string_single_user_message() -> None:
    messages = [{"role": "user", "content": "Just one message."}]
    assert form_prompt_string(messages) == "Just one message."


def test_form_prompt_string_two_user_messages() -> None:
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]
    expected = "User: Hello!\n\n" "Assistant: Hi there!\n\n" "User: How are you?\n\n" "Assistant:"
    assert form_prompt_string(messages) == expected


def test_form_prompt_string_with_system_prompt() -> None:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather?"},
    ]
    expected = "System: You are a helpful assistant.\n\n" "User: What is the weather?\n\n" "Assistant:"
    assert form_prompt_string(messages) == expected


def test_form_prompt_string_missing_content() -> None:
    messages = [
        {"role": "user"},
    ]
    with pytest.raises(KeyError):
        form_prompt_string(messages)


def test_form_prompt_string_warns_on_assistant_last() -> None:
    """Test that a warning is raised when the last message is an assistant message."""
    messages = [
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "content": "Let me check the weather for you."},
    ]
    expected = "User: What's the weather in Paris?\n\n" "Assistant: Let me check the weather for you.\n\n" "Assistant:"
    with pytest.warns(
        UserWarning,
        match="The last message is a tool call or assistant message. The next message should not be an LLM response. "
        "This prompt should not be used for trustworthiness scoring.",
    ):
        assert form_prompt_string(messages) == expected


def test_form_prompt_string_with_tools_chat_completions() -> None:
    """Test formatting with tools in chat completions format."""
    messages = [
        {"role": "user", "content": "What can you do?"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "The search query"}},
                    "required": ["query"],
                },
            },
        }
    ]
    expected = (
        "System: You are an AI Assistant that can call provided tools (a.k.a. functions). "
        "The set of available tools is provided to you as function signatures within "
        "<tools> </tools> XML tags. "
        "You may call one or more of these functions to assist with the user query. If the provided functions are not helpful/relevant, "
        "then just respond in natural conversational language. Don't make assumptions about what values to plug "
        "into functions. After you choose to call a function, you will be provided with the function's results within "
        "<tool_response> </tool_response> XML tags.\n\n"
        "<tools>\n"
        '{"type":"function","function":{"name":"search","description":"Search the web for information","parameters":'
        '{"type":"object","properties":{"query":{"type":"string","description":"The search query"}},"required":["query"]}}}\n'
        "</tools>\n\n"
        "For each function call return a JSON object, with the following pydantic model json schema:\n"
        "{'name': <function-name>, 'arguments': <args-dict>}\n"
        "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
        "Example:\n"
        "<tool_call>\n"
        "{'name': <function-name>, 'arguments': <args-dict>}\n"
        "</tool_call>\n\n"
        "Note: Your past messages will include a call_id in the <tool_call> XML tags. "
        "However, do not generate your own call_id when making a function call.\n\n"
        "User: What can you do?\n\n"
        "Assistant:"
    )
    assert form_prompt_string(messages, tools) == expected


def test_form_prompt_string_with_tools_responses() -> None:
    """Test formatting with tools in responses format."""
    messages = [
        {"role": "user", "content": "What can you do?"},
    ]
    tools = [
        {
            "type": "function",
            "name": "fetch_user_flight_information",
            "description": "Fetch all tickets for the user along with corresponding flight information and seat assignments.\n\n"
            "Returns:\n"
            "    A list of dictionaries where each dictionary contains the ticket details,\n"
            "    associated flight details, and the seat assignments for each ticket belonging to the user.",
            "parameters": {
                "description": "Fetch all tickets for the user along with corresponding flight information and seat assignments.\n\n"
                "Returns:\n"
                "    A list of dictionaries where each dictionary contains the ticket details,\n"
                "    associated flight details, and the seat assignments for each ticket belonging to the user.",
                "properties": {},
                "title": "fetch_user_flight_information",
                "type": "object",
                "additionalProperties": False,
                "required": [],
            },
            "strict": True,
        }
    ]
    expected = (
        "System: You are an AI Assistant that can call provided tools (a.k.a. functions). "
        "The set of available tools is provided to you as function signatures within "
        "<tools> </tools> XML tags. "
        "You may call one or more of these functions to assist with the user query. If the provided functions are not helpful/relevant, "
        "then just respond in natural conversational language. Don't make assumptions about what values to plug "
        "into functions. After you choose to call a function, you will be provided with the function's results within "
        "<tool_response> </tool_response> XML tags.\n\n"
        "<tools>\n"
        '{"type":"function","name":"fetch_user_flight_information","description":"Fetch all tickets for the user along with corresponding flight information and seat assignments.\\n\\n'
        "Returns:\\n"
        "    A list of dictionaries where each dictionary contains the ticket details,\\n"
        '    associated flight details, and the seat assignments for each ticket belonging to the user.","parameters":'
        '{"description":"Fetch all tickets for the user along with corresponding flight information and seat assignments.\\n\\n'
        "Returns:\\n"
        "    A list of dictionaries where each dictionary contains the ticket details,\\n"
        '    associated flight details, and the seat assignments for each ticket belonging to the user.","properties":{},'
        '"title":"fetch_user_flight_information","type":"object","additionalProperties":false,"required":[]},"strict":true}\n'
        "</tools>\n\n"
        "For each function call return a JSON object, with the following pydantic model json schema:\n"
        "{'name': <function-name>, 'arguments': <args-dict>}\n"
        "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
        "Example:\n"
        "<tool_call>\n"
        "{'name': <function-name>, 'arguments': <args-dict>}\n"
        "</tool_call>\n\n"
        "Note: Your past messages will include a call_id in the <tool_call> XML tags. "
        "However, do not generate your own call_id when making a function call.\n\n"
        "User: What can you do?\n\n"
        "Assistant:"
    )
    assert form_prompt_string(messages, tools) == expected


def test_form_prompt_string_with_tool_calls_chat_completions() -> None:
    """Test formatting with tool calls in chat completions format."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_123",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "get_weather",
            "tool_call_id": "call_123",
            "content": "22.1",
        },
    ]
    expected = (
        "User: What's the weather in Paris?\n\n"
        "Assistant: <tool_call>\n"
        "{\n"
        '  "name": "get_weather",\n'
        '  "arguments": {\n'
        '    "location": "Paris"\n'
        "  },\n"
        '  "call_id": "call_123"\n'
        "}\n"
        "</tool_call>\n\n"
        "Tool: "
        "<tool_response>\n"
        "{\n"
        '  "name": "get_weather",\n'
        '  "call_id": "call_123",\n'
        '  "output": "22.1"\n'
        "}\n"
        "</tool_response>\n\n"
        "Assistant:"
    )
    assert form_prompt_string(messages) == expected


def test_form_prompt_string_with_tool_calls_responses() -> None:
    """Test formatting with tool calls in responses format."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "type": "function_call",
            "name": "get_weather",
            "arguments": '{"location": "Paris"}',
            "call_id": "call_123",
        },
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": "22.1",
        },
    ]
    expected = (
        "User: What's the weather in Paris?\n\n"
        "Assistant: <tool_call>\n"
        "{\n"
        '  "name": "get_weather",\n'
        '  "arguments": {\n'
        '    "location": "Paris"\n'
        "  },\n"
        '  "call_id": "call_123"\n'
        "}\n"
        "</tool_call>\n\n"
        "Tool: "
        "<tool_response>\n"
        "{\n"
        '  "name": "get_weather",\n'
        '  "call_id": "call_123",\n'
        '  "output": "22.1"\n'
        "}\n"
        "</tool_response>\n\n"
        "Assistant:"
    )
    assert form_prompt_string(messages) == expected


def test_form_prompt_string_with_tool_calls_two_user_messages_chat_completions() -> None:
    """Test formatting with tool calls and multiple user messages in chat completions format."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_123",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "get_weather",
            "tool_call_id": "call_123",
            "content": "22.1",
        },
        {"role": "assistant", "content": "The temperature in Paris is 22.1°C."},
        {"role": "user", "content": "What about London?"},
    ]
    expected = (
        "User: What's the weather in Paris?\n\n"
        "Assistant: <tool_call>\n"
        "{\n"
        '  "name": "get_weather",\n'
        '  "arguments": {\n'
        '    "location": "Paris"\n'
        "  },\n"
        '  "call_id": "call_123"\n'
        "}\n"
        "</tool_call>\n\n"
        "Tool: "
        "<tool_response>\n"
        "{\n"
        '  "name": "get_weather",\n'
        '  "call_id": "call_123",\n'
        '  "output": "22.1"\n'
        "}\n"
        "</tool_response>\n\n"
        "Assistant: The temperature in Paris is 22.1°C.\n\n"
        "User: What about London?\n\n"
        "Assistant:"
    )
    assert form_prompt_string(messages) == expected


def test_form_prompt_string_with_tool_calls_two_user_messages_responses() -> None:
    """Test formatting with tool calls and multiple user messages in responses format."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "type": "function_call",
            "name": "get_weather",
            "arguments": '{"location": "Paris"}',
            "call_id": "call_123",
        },
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": "22.1",
        },
        {"role": "assistant", "content": "The temperature in Paris is 22.1°C."},
        {"role": "user", "content": "What about London?"},
    ]
    expected = (
        "User: What's the weather in Paris?\n\n"
        "Assistant: <tool_call>\n"
        "{\n"
        '  "name": "get_weather",\n'
        '  "arguments": {\n'
        '    "location": "Paris"\n'
        "  },\n"
        '  "call_id": "call_123"\n'
        "}\n"
        "</tool_call>\n\n"
        "Tool: "
        "<tool_response>\n"
        "{\n"
        '  "name": "get_weather",\n'
        '  "call_id": "call_123",\n'
        '  "output": "22.1"\n'
        "}\n"
        "</tool_response>\n\n"
        "Assistant: The temperature in Paris is 22.1°C.\n\n"
        "User: What about London?\n\n"
        "Assistant:"
    )
    assert form_prompt_string(messages) == expected


def test_form_prompt_string_with_tools_and_system_chat_completions() -> None:
    """Test formatting with tools and system message in chat completions format."""
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are ACME Support, the official AI assistant for ACME Corporation. Your role is to provide exceptional customer service and technical support. You are knowledgeable about all ACME products and services, and you maintain a warm, professional, and solution-oriented approach. You can search our knowledge base to provide accurate and up-to-date information about our products, policies, and support procedures.",
        },
        {"role": "user", "content": "What's the latest news about AI?"},
    ]
    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "The search query"}},
                    "required": ["query"],
                },
            },
        }
    ]
    expected = (
        "System: You are ACME Support, the official AI assistant for ACME Corporation. Your role is to provide exceptional customer service and technical support. You are knowledgeable about all ACME products and services, and you maintain a warm, professional, and solution-oriented approach. You can search our knowledge base to provide accurate and up-to-date information about our products, policies, and support procedures.\n\n"
        "You are an AI Assistant that can call provided tools (a.k.a. functions). "
        "The set of available tools is provided to you as function signatures within "
        "<tools> </tools> XML tags. "
        "You may call one or more of these functions to assist with the user query. If the provided functions are not helpful/relevant, "
        "then just respond in natural conversational language. Don't make assumptions about what values to plug "
        "into functions. After you choose to call a function, you will be provided with the function's results within "
        "<tool_response> </tool_response> XML tags.\n\n"
        "<tools>\n"
        '{"type":"function","function":{"name":"search","description":"Search the web for information","parameters":'
        '{"type":"object","properties":{"query":{"type":"string","description":"The search query"}},"required":["query"]}}}\n'
        "</tools>\n\n"
        "For each function call return a JSON object, with the following pydantic model json schema:\n"
        "{'name': <function-name>, 'arguments': <args-dict>}\n"
        "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
        "Example:\n"
        "<tool_call>\n"
        "{'name': <function-name>, 'arguments': <args-dict>}\n"
        "</tool_call>\n\n"
        "Note: Your past messages will include a call_id in the <tool_call> XML tags. "
        "However, do not generate your own call_id when making a function call.\n\n"
        "User: What's the latest news about AI?\n\n"
        "Assistant:"
    )
    assert form_prompt_string(messages, tools) == expected


def test_form_prompt_string_with_tools_and_system_responses() -> None:
    """Test formatting with tools and system message in responses format."""
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are ACME Support, the official AI assistant for ACME Corporation. Your role is to provide exceptional customer service and technical support. You are knowledgeable about all ACME products and services, and you maintain a warm, professional, and solution-oriented approach. You can search our knowledge base to provide accurate and up-to-date information about our products, policies, and support procedures.",
        },
        {"role": "user", "content": "What's the latest news about AI?"},
    ]
    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "name": "search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query"}},
                "required": ["query"],
            },
            "strict": True,
        }
    ]
    expected = (
        "System: You are ACME Support, the official AI assistant for ACME Corporation. Your role is to provide exceptional customer service and technical support. You are knowledgeable about all ACME products and services, and you maintain a warm, professional, and solution-oriented approach. You can search our knowledge base to provide accurate and up-to-date information about our products, policies, and support procedures.\n\n"
        "You are an AI Assistant that can call provided tools (a.k.a. functions). "
        "The set of available tools is provided to you as function signatures within "
        "<tools> </tools> XML tags. "
        "You may call one or more of these functions to assist with the user query. If the provided functions are not helpful/relevant, "
        "then just respond in natural conversational language. Don't make assumptions about what values to plug "
        "into functions. After you choose to call a function, you will be provided with the function's results within "
        "<tool_response> </tool_response> XML tags.\n\n"
        "<tools>\n"
        '{"type":"function","name":"search","description":"Search the web for information","parameters":'
        '{"type":"object","properties":{"query":{"type":"string","description":"The search query"}},"required":["query"]},'
        '"strict":true}\n'
        "</tools>\n\n"
        "For each function call return a JSON object, with the following pydantic model json schema:\n"
        "{'name': <function-name>, 'arguments': <args-dict>}\n"
        "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
        "Example:\n"
        "<tool_call>\n"
        "{'name': <function-name>, 'arguments': <args-dict>}\n"
        "</tool_call>\n\n"
        "Note: Your past messages will include a call_id in the <tool_call> XML tags. "
        "However, do not generate your own call_id when making a function call.\n\n"
        "User: What's the latest news about AI?\n\n"
        "Assistant:"
    )
    assert form_prompt_string(messages, tools) == expected


def test_form_prompt_string_warns_on_tool_call_last_chat_completions() -> None:
    """Test that a warning is raised when the last message is a tool call in chat completions format."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_123",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}',
                    },
                }
            ],
        },
    ]
    expected = (
        "User: What's the weather in Paris?\n\n"
        "Assistant: <tool_call>\n"
        "{\n"
        '  "name": "get_weather",\n'
        '  "arguments": {\n'
        '    "location": "Paris"\n'
        "  },\n"
        '  "call_id": "call_123"\n'
        "}\n"
        "</tool_call>\n\n"
        "Assistant:"
    )
    with pytest.warns(
        UserWarning,
        match="The last message is a tool call or assistant message. The next message should not be an LLM response. "
        "This prompt should not be used for trustworthiness scoring.",
    ):
        assert form_prompt_string(messages) == expected


def test_form_prompt_string_warns_on_tool_call_last_responses() -> None:
    """Test that a warning is raised when the last message is a tool call in responses format."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "type": "function_call",
            "name": "get_weather",
            "arguments": '{"location": "Paris"}',
            "call_id": "call_123",
        },
    ]
    expected = (
        "User: What's the weather in Paris?\n\n"
        "Assistant: <tool_call>\n"
        "{\n"
        '  "name": "get_weather",\n'
        '  "arguments": {\n'
        '    "location": "Paris"\n'
        "  },\n"
        '  "call_id": "call_123"\n'
        "}\n"
        "</tool_call>\n\n"
        "Assistant:"
    )
    with pytest.warns(
        UserWarning,
        match="The last message is a tool call or assistant message. The next message should not be an LLM response. "
        "This prompt should not be used for trustworthiness scoring.",
    ):
        assert form_prompt_string(messages) == expected

    """Test that form_prompt_string correctly handles tools in the Responses API format."""
    responses_tools = [
        {
            "type": "function",
            "name": "fetch_user_flight_information",
            "description": "Fetch flight information",
            "parameters": {
                "description": "Fetch flight information",
                "properties": {},
                "title": "fetch_user_flight_information",
                "type": "object",
                "additionalProperties": False,
                "required": [],
            },
            "strict": True,
        }
    ]
    responses_tools_expected = (
        "System: You are an AI Assistant that can call provided tools (a.k.a. functions). "
        "The set of available tools is provided to you as function signatures within "
        "<tools> </tools> XML tags. "
        "You may call one or more of these functions to assist with the user query. If the provided functions are not helpful/relevant, "
        "then just respond in natural conversational language. Don't make assumptions about what values to plug "
        "into functions. After you choose to call a function, you will be provided with the function's results within "
        "<tool_response> </tool_response> XML tags.\n\n"
        "<tools>\n"
        '{"type":"function","name":"fetch_user_flight_information","description":"Fetch flight information","parameters":'
        '{"description":"Fetch flight information","properties":{},"title":"fetch_user_flight_information","type":"object",'
        '"additionalProperties":false,"required":[]},"strict":true}\n'
        "</tools>\n\n"
        "For each function call return a JSON object, with the following pydantic model json schema:\n"
        "{'name': <function-name>, 'arguments': <args-dict>}\n"
        "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
        "Example:\n"
        "<tool_call>\n"
        "{'name': <function-name>, 'arguments': <args-dict>}\n"
        "</tool_call>\n\n"
        "Note: Your past messages will include a call_id in the <tool_call> XML tags. "
        "However, do not generate your own call_id when making a function call.\n\n"
        "User: What can you do?\n\n"
        "Assistant:"
    )
    assert (
        form_prompt_string([{"role": "user", "content": "What can you do?"}], responses_tools)
        == responses_tools_expected
    )


def test_form_prompt_string_assistant_content_before_tool_calls_chat_completions() -> None:
    """Test that assistant messages with both content and tool calls have content before tool calls in chat completions format."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Can you help me find information about ACME's warranty policy?"},
        {
            "role": "assistant",
            "content": "I'll help you find information about our warranty policy. Let me search our knowledge base for the details.",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_123",
                    "function": {
                        "name": "search_knowledge_base",
                        "arguments": '{"query": "ACME warranty policy terms and conditions"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "search_knowledge_base",
            "tool_call_id": "call_123",
            "content": "ACME offers a 2-year warranty on all products. The warranty covers manufacturing defects and normal wear and tear.",
        },
    ]
    expected = (
        "User: Can you help me find information about ACME's warranty policy?\n\n"
        "Assistant: I'll help you find information about our warranty policy. Let me search our knowledge base for the details.\n\n"
        "<tool_call>\n"
        "{\n"
        '  "name": "search_knowledge_base",\n'
        '  "arguments": {\n'
        '    "query": "ACME warranty policy terms and conditions"\n'
        "  },\n"
        '  "call_id": "call_123"\n'
        "}\n"
        "</tool_call>\n\n"
        "Tool: "
        "<tool_response>\n"
        "{\n"
        '  "name": "search_knowledge_base",\n'
        '  "call_id": "call_123",\n'
        '  "output": "ACME offers a 2-year warranty on all products. The warranty covers manufacturing defects and normal wear and tear."\n'
        "}\n"
        "</tool_response>\n\n"
        "Assistant:"
    )
    assert form_prompt_string(messages) == expected


def test_form_prompt_string_assistant_content_before_tool_calls_responses() -> None:
    """Test that assistant messages with both content and tool calls have content before tool calls in responses format."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Can you help me find information about ACME's warranty policy?"},
        {
            "type": "function_call",
            "name": "search_knowledge_base",
            "arguments": '{"query": "ACME warranty policy terms and conditions"}',
            "call_id": "call_123",
            "content": "I'll help you find information about our warranty policy. Let me search our knowledge base for the details.",
        },
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": "ACME offers a 2-year warranty on all products. The warranty covers manufacturing defects and normal wear and tear.",
        },
    ]
    expected = (
        "User: Can you help me find information about ACME's warranty policy?\n\n"
        "Assistant: I'll help you find information about our warranty policy. Let me search our knowledge base for the details.\n\n"
        "<tool_call>\n"
        "{\n"
        '  "name": "search_knowledge_base",\n'
        '  "arguments": {\n'
        '    "query": "ACME warranty policy terms and conditions"\n'
        "  },\n"
        '  "call_id": "call_123"\n'
        "}\n"
        "</tool_call>\n\n"
        "Tool: "
        "<tool_response>\n"
        "{\n"
        '  "name": "search_knowledge_base",\n'
        '  "call_id": "call_123",\n'
        '  "output": "ACME offers a 2-year warranty on all products. The warranty covers manufacturing defects and normal wear and tear."\n'
        "}\n"
        "</tool_response>\n\n"
        "Assistant:"
    )
    assert form_prompt_string(messages) == expected


def test_form_prompt_string_with_instructions_responses() -> None:
    """Test formatting with developer instructions in responses format."""
    messages = [
        {"role": "user", "content": "What can you do?"},
    ]
    expected = "System: Always be concise and direct in your responses.\n\n" "User: What can you do?\n\n" "Assistant:"
    assert form_prompt_string(messages, instructions="Always be concise and direct in your responses.") == expected


def test_form_prompt_string_with_instructions_and_tools_responses() -> None:
    """Test formatting with developer instructions and tools in responses format."""
    messages = [
        {"role": "user", "content": "What can you do?"},
    ]
    tools = [
        {
            "type": "function",
            "name": "search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query"}},
                "required": ["query"],
            },
            "strict": True,
        }
    ]
    expected = (
        "System: Always be concise and direct in your responses.\n\n"
        "You are an AI Assistant that can call provided tools (a.k.a. functions). "
        "The set of available tools is provided to you as function signatures within "
        "<tools> </tools> XML tags. "
        "You may call one or more of these functions to assist with the user query. If the provided functions are not helpful/relevant, "
        "then just respond in natural conversational language. Don't make assumptions about what values to plug "
        "into functions. After you choose to call a function, you will be provided with the function's results within "
        "<tool_response> </tool_response> XML tags.\n\n"
        "<tools>\n"
        '{"type":"function","name":"search","description":"Search the web for information","parameters":'
        '{"type":"object","properties":{"query":{"type":"string","description":"The search query"}},"required":["query"]},'
        '"strict":true}\n'
        "</tools>\n\n"
        "For each function call return a JSON object, with the following pydantic model json schema:\n"
        "{'name': <function-name>, 'arguments': <args-dict>}\n"
        "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
        "Example:\n"
        "<tool_call>\n"
        "{'name': <function-name>, 'arguments': <args-dict>}\n"
        "</tool_call>\n\n"
        "Note: Your past messages will include a call_id in the <tool_call> XML tags. "
        "However, do not generate your own call_id when making a function call.\n\n"
        "User: What can you do?\n\n"
        "Assistant:"
    )
    assert (
        form_prompt_string(messages, tools=tools, instructions="Always be concise and direct in your responses.")
        == expected
    )


def test_form_prompt_string_with_instructions_and_tool_calls_responses() -> None:
    """Test formatting with developer instructions and tool calls in responses format."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "type": "function_call",
            "name": "get_weather",
            "arguments": '{"location": "Paris"}',
            "call_id": "call_123",
        },
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": "22.1",
        },
    ]
    expected = (
        "System: Always be concise and direct in your responses.\n\n"
        "User: What's the weather in Paris?\n\n"
        "Assistant: <tool_call>\n"
        "{\n"
        '  "name": "get_weather",\n'
        '  "arguments": {\n'
        '    "location": "Paris"\n'
        "  },\n"
        '  "call_id": "call_123"\n'
        "}\n"
        "</tool_call>\n\n"
        "Tool: "
        "<tool_response>\n"
        "{\n"
        '  "name": "get_weather",\n'
        '  "call_id": "call_123",\n'
        '  "output": "22.1"\n'
        "}\n"
        "</tool_response>\n\n"
        "Assistant:"
    )
    assert form_prompt_string(messages, instructions="Always be concise and direct in your responses.") == expected


def test_form_prompt_string_with_instructions_chat_completions_throws_error() -> None:
    """Test that Responses API parameters cannot be used with use_responses=False."""
    messages = [
        {"role": "user", "content": "What can you do?"},
    ]
    with pytest.raises(
        ValueError,
        match="Responses API kwargs are only supported in Responses API format. Cannot use with use_responses=False.",
    ):
        form_prompt_string(
            messages, instructions="Always be concise and direct in your responses.", use_responses=False
        )


def test_form_prompt_string_with_developer_role_begin() -> None:
    """Test formatting with developer role in the beginning of the messages list."""
    messages = [
        {"role": "developer", "content": "Always be concise and direct in your responses."},
        {"role": "user", "content": "What can you do?"},
    ]
    expected = "System: Always be concise and direct in your responses.\n\n" "User: What can you do?\n\n" "Assistant:"
    assert form_prompt_string(messages) == expected


def test_form_prompt_string_with_developer_role_middle() -> None:
    """Test formatting with developer role in the middle of the messages list."""
    messages = [
        {"role": "user", "content": "What can you do?"},
        {"role": "developer", "content": "Always be concise and direct in your responses."},
    ]
    expected = "User: What can you do?\n\n" "System: Always be concise and direct in your responses.\n\n" "Assistant:"
    assert form_prompt_string(messages) == expected


def test_form_prompt_string_with_developer_role_and_tools() -> None:
    """Test formatting with developer role and tool list."""
    messages = [
        {"role": "developer", "content": "Always be concise and direct in your responses."},
        {"role": "user", "content": "What can you do?"},
    ]
    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "name": "search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query"}},
                "required": ["query"],
            },
            "strict": True,
        }
    ]
    expected = (
        "System: Always be concise and direct in your responses.\n\n"
        "You are an AI Assistant that can call provided tools (a.k.a. functions). "
        "The set of available tools is provided to you as function signatures within "
        "<tools> </tools> XML tags. "
        "You may call one or more of these functions to assist with the user query. If the provided functions are not helpful/relevant, "
        "then just respond in natural conversational language. Don't make assumptions about what values to plug "
        "into functions. After you choose to call a function, you will be provided with the function's results within "
        "<tool_response> </tool_response> XML tags.\n\n"
        "<tools>\n"
        '{"type":"function","name":"search","description":"Search the web for information","parameters":'
        '{"type":"object","properties":{"query":{"type":"string","description":"The search query"}},"required":["query"]},'
        '"strict":true}\n'
        "</tools>\n\n"
        "For each function call return a JSON object, with the following pydantic model json schema:\n"
        "{'name': <function-name>, 'arguments': <args-dict>}\n"
        "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
        "Example:\n"
        "<tool_call>\n"
        "{'name': <function-name>, 'arguments': <args-dict>}\n"
        "</tool_call>\n\n"
        "Note: Your past messages will include a call_id in the <tool_call> XML tags. "
        "However, do not generate your own call_id when making a function call.\n\n"
        "User: What can you do?\n\n"
        "Assistant:"
    )
    assert form_prompt_string(messages, tools=tools) == expected


def test_form_prompt_string_with_instructions_developer_role_and_tools() -> None:
    """Test formatting with instructions, developer role and tool list."""
    messages = [
        {"role": "developer", "content": "Always be concise and direct in your responses."},
        {"role": "user", "content": "What can you do?"},
    ]
    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "name": "search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query"}},
                "required": ["query"],
            },
            "strict": True,
        }
    ]
    expected = (
        "System: This system prompt appears first.\n\n"
        "Always be concise and direct in your responses.\n\n"
        "You are an AI Assistant that can call provided tools (a.k.a. functions). "
        "The set of available tools is provided to you as function signatures within "
        "<tools> </tools> XML tags. "
        "You may call one or more of these functions to assist with the user query. If the provided functions are not helpful/relevant, "
        "then just respond in natural conversational language. Don't make assumptions about what values to plug "
        "into functions. After you choose to call a function, you will be provided with the function's results within "
        "<tool_response> </tool_response> XML tags.\n\n"
        "<tools>\n"
        '{"type":"function","name":"search","description":"Search the web for information","parameters":'
        '{"type":"object","properties":{"query":{"type":"string","description":"The search query"}},"required":["query"]},'
        '"strict":true}\n'
        "</tools>\n\n"
        "For each function call return a JSON object, with the following pydantic model json schema:\n"
        "{'name': <function-name>, 'arguments': <args-dict>}\n"
        "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
        "Example:\n"
        "<tool_call>\n"
        "{'name': <function-name>, 'arguments': <args-dict>}\n"
        "</tool_call>\n\n"
        "Note: Your past messages will include a call_id in the <tool_call> XML tags. "
        "However, do not generate your own call_id when making a function call.\n\n"
        "User: What can you do?\n\n"
        "Assistant:"
    )
    assert form_prompt_string(messages, tools=tools, instructions="This system prompt appears first.") == expected


@pytest.mark.parametrize("use_tools", [False, True])
@pytest.mark.filterwarnings("ignore:The last message is a tool call or assistant message")
def test_form_prompt_string_does_not_mutate_messages(use_tools: bool) -> None:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_capital",
                "description": "Get the capital of a country",
                "parameters": {"type": "object", "properties": {"country": {"type": "string"}}},
            },
        },
    ]

    original_messages = [dict(msg) for msg in messages]
    original_len = len(messages)

    form_prompt_string(messages=messages, tools=tools if use_tools else None)

    # Verify length hasn't changed
    assert len(messages) == original_len, (
        f"form_prompt_string mutated messages: " f"expected length {original_len}, got {len(messages)}"
    )

    # Verify message contents haven't changed
    for original, current in zip(original_messages, messages):
        assert current == original, (
            f"form_prompt_string mutated message content: " f"expected {original}, got {current}"
        )


@pytest.mark.parametrize("use_tools", [False, True])
@pytest.mark.filterwarnings("ignore:The last message is a tool call or assistant message")
def test_form_prompt_chat_completions_api_does_not_mutate_messages(use_tools: bool) -> None:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_capital",
                "description": "Get the capital of a country",
                "parameters": {"type": "object", "properties": {"country": {"type": "string"}}},
            },
        },
    ]

    original_messages = [dict(msg) for msg in messages]
    original_len = len(messages)

    _form_prompt_chat_completions_api(
        messages=cast(list["ChatCompletionMessageParam"], messages), tools=tools if use_tools else None
    )

    # Verify length hasn't changed
    assert len(messages) == original_len, (
        f"_form_prompt_chat_completions_api mutated messages: " f"expected length {original_len}, got {len(messages)}"
    )

    # Verify message contents haven't changed
    for original, current in zip(original_messages, messages):
        assert current == original, (
            f"_form_prompt_chat_completions_api mutated message content: " f"expected {original}, got {current}"
        )


@pytest.mark.parametrize("use_tools", [False, True])
@pytest.mark.filterwarnings("ignore:The last message is a tool call or assistant message")
def test_form_prompt_responses_api_does_not_mutate_messages(use_tools: bool) -> None:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris"},
    ]
    tools = [
        {
            "type": "function",
            "name": "get_capital",
            "description": "Get the capital of a country",
            "parameters": {"type": "object", "properties": {"country": {"type": "string"}}},
        },
    ]

    original_messages = [dict(msg) for msg in messages]
    original_len = len(messages)

    _form_prompt_responses_api(messages=messages, tools=tools if use_tools else None)

    # Verify length hasn't changed
    assert len(messages) == original_len, (
        f"_form_prompt_responses_api mutated messages: " f"expected length {original_len}, got {len(messages)}"
    )

    # Verify message contents haven't changed
    for original, current in zip(original_messages, messages):
        assert current == original, (
            f"_form_prompt_responses_api mutated message content: " f"expected {original}, got {current}"
        )


@pytest.mark.parametrize("use_responses", [False, True])
def test_form_prompt_string_with_tools_after_first_system_block(use_responses: bool) -> None:
    """Test that tools are inserted after the first consecutive block of system messages in both formats."""
    messages = [
        {"role": "system", "content": "First system message."},
        {"role": "system", "content": "Second system message."},
        {"role": "user", "content": "What can you do?"},
        {"role": "assistant", "content": "I can help you."},
        {"role": "system", "content": "Third system message later."},
        {"role": "user", "content": "Tell me more."},
    ]

    if use_responses:
        # Responses format includes strict field
        tools = [
            {
                "type": "function",
                "name": "search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "The search query"}},
                    "required": ["query"],
                },
                "strict": True,
            }
        ]
        expected = (
            "System: First system message.\n\n"
            "Second system message.\n\n"
            "You are an AI Assistant that can call provided tools (a.k.a. functions). "
            "The set of available tools is provided to you as function signatures within "
            "<tools> </tools> XML tags. "
            "You may call one or more of these functions to assist with the user query. If the provided functions are not helpful/relevant, "
            "then just respond in natural conversational language. Don't make assumptions about what values to plug "
            "into functions. After you choose to call a function, you will be provided with the function's results within "
            "<tool_response> </tool_response> XML tags.\n\n"
            "<tools>\n"
            '{"type":"function","name":"search","description":"Search the web for information","parameters":'
            '{"type":"object","properties":{"query":{"type":"string","description":"The search query"}},"required":["query"]},'
            '"strict":true}\n'
            "</tools>\n\n"
            "For each function call return a JSON object, with the following pydantic model json schema:\n"
            "{'name': <function-name>, 'arguments': <args-dict>}\n"
            "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
            "Example:\n"
            "<tool_call>\n"
            "{'name': <function-name>, 'arguments': <args-dict>}\n"
            "</tool_call>\n\n"
            "Note: Your past messages will include a call_id in the <tool_call> XML tags. "
            "However, do not generate your own call_id when making a function call.\n\n"
            "User: What can you do?\n\n"
            "Assistant: I can help you.\n\n"
            "System: Third system message later.\n\n"
            "User: Tell me more.\n\n"
            "Assistant:"
        )
    else:
        # Chat completions format uses nested function structure
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string", "description": "The search query"}},
                        "required": ["query"],
                    },
                },
            }
        ]
        expected = (
            "System: First system message.\n\n"
            "Second system message.\n\n"
            "You are an AI Assistant that can call provided tools (a.k.a. functions). "
            "The set of available tools is provided to you as function signatures within "
            "<tools> </tools> XML tags. "
            "You may call one or more of these functions to assist with the user query. If the provided functions are not helpful/relevant, "
            "then just respond in natural conversational language. Don't make assumptions about what values to plug "
            "into functions. After you choose to call a function, you will be provided with the function's results within "
            "<tool_response> </tool_response> XML tags.\n\n"
            "<tools>\n"
            '{"type":"function","function":{"name":"search","description":"Search the web for information","parameters":'
            '{"type":"object","properties":{"query":{"type":"string","description":"The search query"}},"required":["query"]}}}\n'
            "</tools>\n\n"
            "For each function call return a JSON object, with the following pydantic model json schema:\n"
            "{'name': <function-name>, 'arguments': <args-dict>}\n"
            "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
            "Example:\n"
            "<tool_call>\n"
            "{'name': <function-name>, 'arguments': <args-dict>}\n"
            "</tool_call>\n\n"
            "Note: Your past messages will include a call_id in the <tool_call> XML tags. "
            "However, do not generate your own call_id when making a function call.\n\n"
            "User: What can you do?\n\n"
            "Assistant: I can help you.\n\n"
            "System: Third system message later.\n\n"
            "User: Tell me more.\n\n"
            "Assistant:"
        )

    result = form_prompt_string(messages, tools, use_responses=use_responses)
    assert result == expected


@pytest.mark.parametrize("use_responses", [False, True])
def test_form_prompt_string_with_empty_tools(use_responses: bool) -> None:
    """Test that empty tools list is treated the same as None in both formats."""
    messages = [{"role": "user", "content": "Just one message."}]

    # Test with None
    result_none = form_prompt_string(messages, tools=None, use_responses=use_responses)

    # Test with empty list
    result_empty = form_prompt_string(messages, tools=[], use_responses=use_responses)

    # They should be identical
    assert result_none == result_empty == "Just one message."


@pytest.mark.parametrize("use_responses", [False, True])
def test_form_prompt_string_with_empty_tools_multiple_messages(use_responses: bool) -> None:
    """Test empty tools list with multiple messages in both formats."""
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]

    # Test with None
    result_none = form_prompt_string(messages, tools=None, use_responses=use_responses)

    # Test with empty list
    result_empty = form_prompt_string(messages, tools=[], use_responses=use_responses)

    # They should be identical
    expected = "User: Hello!\n\n" "Assistant: Hi there!\n\n" "User: How are you?\n\n" "Assistant:"
    assert result_none == result_empty == expected


@pytest.mark.parametrize("use_responses", [False, True])
def test_form_prompt_string_with_empty_arguments(use_responses: bool) -> None:
    """Test formatting with tool calls having empty arguments string in both formats."""
    if use_responses:
        # Responses format
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Execute the action"},
            {
                "type": "function_call",
                "name": "execute_action",
                "arguments": "",
                "call_id": "call_123",
            },
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "Action completed successfully",
            },
        ]
    else:
        # Chat completions format
        messages = [
            {"role": "user", "content": "Execute the action"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_123",
                        "function": {
                            "name": "execute_action",
                            "arguments": "",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "execute_action",
                "tool_call_id": "call_123",
                "content": "Action completed successfully",
            },
        ]

    expected = (
        "User: Execute the action\n\n"
        "Assistant: <tool_call>\n"
        "{\n"
        '  "name": "execute_action",\n'
        '  "arguments": {},\n'
        '  "call_id": "call_123"\n'
        "}\n"
        "</tool_call>\n\n"
        "Tool: "
        "<tool_response>\n"
        "{\n"
        '  "name": "execute_action",\n'
        '  "call_id": "call_123",\n'
        '  "output": "Action completed successfully"\n'
        "}\n"
        "</tool_response>\n\n"
        "Assistant:"
    )
    assert form_prompt_string(messages, use_responses=use_responses) == expected


def test_form_response_string_chat_completions_api_just_content() -> None:
    """Test form_response_string_chat_completions_api with just content."""
    response = {"content": "Hello, how can I help you today?"}
    expected = "Hello, how can I help you today?"
    result = form_response_string_chat_completions_api(response)
    assert result == expected


def test_form_response_string_chat_completions_api_just_tool_calls() -> None:
    """Test form_response_string_chat_completions_api with just tool calls."""
    response = {
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Paris"}',
                }
            }
        ],
    }
    expected = (
        "<tool_call>\n"
        "{\n"
        '  "name": "get_weather",\n'
        '  "arguments": {\n'
        '    "location": "Paris"\n'
        "  }\n"
        "}\n"
        "</tool_call>"
    )
    result = form_response_string_chat_completions_api(response)
    assert result == expected


def test_form_response_string_chat_completions_api_content_and_tool_calls() -> None:
    """Test form_response_string_chat_completions_api with both content and tool calls."""
    response = {
        "role": "assistant",
        "content": "I'll check the weather for you.",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Paris"}',
                }
            }
        ],
    }
    expected = (
        "I'll check the weather for you.\n"
        "<tool_call>\n"
        "{\n"
        '  "name": "get_weather",\n'
        '  "arguments": {\n'
        '    "location": "Paris"\n'
        "  }\n"
        "}\n"
        "</tool_call>"
    )
    result = form_response_string_chat_completions_api(response)
    assert result == expected


def test_form_response_string_chat_completions_api_multiple_tool_calls() -> None:
    """Test form_response_string_chat_completions_api with multiple tool calls."""
    response = {
        "role": "assistant",
        "content": "Let me check multiple things for you.",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Paris"}',
                }
            },
            {
                "function": {
                    "name": "get_time",
                    "arguments": '{"timezone": "UTC"}',
                }
            },
        ],
    }
    expected = (
        "Let me check multiple things for you.\n"
        "<tool_call>\n"
        "{\n"
        '  "name": "get_weather",\n'
        '  "arguments": {\n'
        '    "location": "Paris"\n'
        "  }\n"
        "}\n"
        "</tool_call>\n"
        "<tool_call>\n"
        "{\n"
        '  "name": "get_time",\n'
        '  "arguments": {\n'
        '    "timezone": "UTC"\n'
        "  }\n"
        "}\n"
        "</tool_call>"
    )
    result = form_response_string_chat_completions_api(response)
    assert result == expected


def test_form_response_string_chat_completions_api_empty_content() -> None:
    """Test form_response_string_chat_completions_api with empty content."""
    response = {"content": ""}
    expected = ""
    result = form_response_string_chat_completions_api(response)
    assert result == expected


def test_form_response_string_chat_completions_api_missing_content() -> None:
    """Test form_response_string_chat_completions_api with missing content key."""
    response: dict[str, Any] = {}
    expected = ""
    result = form_response_string_chat_completions_api(response)
    assert result == expected


def test_form_response_string_chat_completions_api_empty_arguments() -> None:
    """Test form_response_string_chat_completions_api with empty arguments."""
    response = {
        "role": "assistant",
        "content": "Running action",
        "tool_calls": [
            {
                "function": {
                    "name": "execute_action",
                    "arguments": "",
                }
            }
        ],
    }
    expected = (
        "Running action\n"
        "<tool_call>\n"
        "{\n"
        '  "name": "execute_action",\n'
        '  "arguments": {}\n'
        "}\n"
        "</tool_call>"
    )
    result = form_response_string_chat_completions_api(response)
    assert result == expected


def test_form_response_string_chat_completions_api_invalid_input() -> None:
    """Test form_response_string_chat_completions_api raises TypeError for invalid input."""
    with pytest.raises(TypeError, match="Expected response to be a dict, got str"):
        form_response_string_chat_completions_api("not a dict")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Expected response to be a dict, got list"):
        form_response_string_chat_completions_api([])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Expected response to be a dict, got NoneType"):
        form_response_string_chat_completions_api(None)  # type: ignore[arg-type]


def test_form_response_string_chat_completions_api_malformed_tool_calls() -> None:
    """Test form_response_string_chat_completions_api handles malformed tool calls gracefully."""
    # Test with missing function key - this should trigger a warning
    response = {
        "role": "assistant",
        "content": "I'll help you.",
        "tool_calls": [{"invalid": "structure"}],
    }

    with pytest.warns(UserWarning, match="Error formatting tool_calls in response: 'function'"):
        result = form_response_string_chat_completions_api(response)
        assert result == "I'll help you."

    # Test with invalid JSON in arguments - this should trigger a warning
    response = {
        "content": "Let me check that.",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": "invalid json{",
                }
            }
        ],
    }

    # Warning expected since JSON parsing will fail
    with pytest.warns(UserWarning, match="Error formatting tool_calls in response.*Returning content only"):
        result = form_response_string_chat_completions_api(response)
        assert result == "Let me check that."

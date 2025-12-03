from typing import TYPE_CHECKING, Any, cast

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.responses import Response
from openai.types.responses.file_search_tool import FileSearchTool, RankingOptions
from openai.types.responses.response_file_search_tool_call import (
    ResponseFileSearchToolCall,
    Result,
)
from openai.types.responses.response_function_web_search import (
    ActionSearch,
    ActionSearchSource,
    ResponseFunctionWebSearch,
)
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import (
    AnnotationFileCitation,
    AnnotationURLCitation,
    ResponseOutputText,
)
from openai.types.responses.tool_choice_types import ToolChoiceTypes
from openai.types.responses.web_search_tool import UserLocation, WebSearchTool

from cleanlab_tlm.internal.rag import _is_tool_call_response
from cleanlab_tlm.utils.chat import (
    _form_prompt_chat_completions_api,
    _form_prompt_responses_api,
    form_prompt_string,
    form_response_string_chat_completions,
    form_response_string_chat_completions_api,
    form_response_string_responses_api,
)
from tests.openai_compat import ChatCompletionMessageToolCall, Function

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

####################### FORM_PROMPT_STRING ##############################
# Tests for the form_prompt_string function which uses either Responses
# API or Chat Completions API.
#########################################################################


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
        "then just respond in natural conversational language. "
        "After you choose to call a function, you will be provided with the function's results within "
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
        "</tool_call>\n"
        "Note: Function calls and their results may optionally include a call_id, which should be ignored.\n\n"
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
        "then just respond in natural conversational language. "
        "After you choose to call a function, you will be provided with the function's results within "
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
        "</tool_call>\n"
        "Note: Function calls and their results may optionally include a call_id, which should be ignored.\n\n"
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
        "then just respond in natural conversational language. "
        "After you choose to call a function, you will be provided with the function's results within "
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
        "</tool_call>\n"
        "Note: Function calls and their results may optionally include a call_id, which should be ignored.\n\n"
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
        "then just respond in natural conversational language. "
        "After you choose to call a function, you will be provided with the function's results within "
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
        "</tool_call>\n"
        "Note: Function calls and their results may optionally include a call_id, which should be ignored.\n\n"
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
        "then just respond in natural conversational language. "
        "After you choose to call a function, you will be provided with the function's results within "
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
        "</tool_call>\n"
        "Note: Function calls and their results may optionally include a call_id, which should be ignored.\n\n"
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
        "then just respond in natural conversational language. "
        "After you choose to call a function, you will be provided with the function's results within "
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
        "</tool_call>\n"
        "Note: Function calls and their results may optionally include a call_id, which should be ignored.\n\n"
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
        "then just respond in natural conversational language. "
        "After you choose to call a function, you will be provided with the function's results within "
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
        "</tool_call>\n"
        "Note: Function calls and their results may optionally include a call_id, which should be ignored.\n\n"
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
        "then just respond in natural conversational language. "
        "After you choose to call a function, you will be provided with the function's results within "
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
        "</tool_call>\n"
        "Note: Function calls and their results may optionally include a call_id, which should be ignored.\n\n"
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
            "then just respond in natural conversational language. "
            "After you choose to call a function, you will be provided with the function's results within "
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
            "</tool_call>\n"
            "Note: Function calls and their results may optionally include a call_id, which should be ignored.\n\n"
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
            "then just respond in natural conversational language. "
            "After you choose to call a function, you will be provided with the function's results within "
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
            "</tool_call>\n"
            "Note: Function calls and their results may optionally include a call_id, which should be ignored.\n\n"
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


########### FORM_PROMPT_STRING (Responses API kwargs) ###################
# Tests for the form_prompt_string function using Responses API kwargs.
#########################################################################


def test_form_prompt_string_responses_file_search() -> None:
    """Test form prompt string in OpenAI Responses with File Search."""
    openai_kwargs = {
        "tools": [{"type": "file_search", "vector_store_ids": ["ID"]}],
        "include": ["file_search_call.results"],
        "model": "gpt-4.1-mini",
        "input": "How much gpt-5 cost?",
    }

    response = Response(
        id="resp_68b0ef38417481929bcf0b9cb8f884120c6a831e0341a521",
        created_at=1756426040.0,
        model="gpt-4.1-mini-2025-04-14",
        object="response",
        output=[
            ResponseFileSearchToolCall(
                id="fs_68b0ef38cc0081928e29c0a0a29f8bb30c6a831e0341a521",
                queries=["cost of GPT-5", "price of GPT-5", "GPT-5 cost"],
                status="completed",
                type="file_search_call",
                results=[
                    Result(
                        attributes={},
                        file_id="file-K6W51Znon7LxKCv8B6AgoE",
                        filename="openai-pricing.pdf",
                        score=0.9195,
                        text="Pricing | OpenAI\n\n\nPricing below reflects standard processing rates. To optimize cost and performance for\n\ndifferent use cases, we also offer:\n\nBatch API  : Save 50% on inputs and outputs with the Batch API and run tasks\n\nasynchronously over 24 hours.\n\nPriority processing  : offers reliable, high-speed performance with the flexibility to\n\npay-as-you-go.\n\nGPT-5\n\nThe best model for coding and agentic tasks across industries\n\nPrice\n\nAPI Pricing\n\nContact sales\n\nFlagship models\n\nOur frontier models designed to spend more time thinking before\n\nproducing a response, making them ideal for complex, multi-step problems.\n\n8/21/25, 8:18 PM Pricing | OpenAI\n\nhttps://openai.com/api/pricing/ 1/11\n\nhttps://platform.openai.com/docs/guides/batch\nhttps://openai.com/api-priority-processing/\nhttps://openai.com/contact-sales/\nhttps://openai.com/\n\n\nInput:\n\n$1.250 / 1M tokens\n\nCached input:\n\n$0.125 / 1M tokens\n\nOutput:\n\n$10.000 / 1M tokens\n\nGPT-5 mini\nA faster, cheaper version of GPT-5 for well-defined tasks\n\nPrice\n\nInput:\n\n$0.250 / 1M tokens\n\nCached input:\n\n$0.025 / 1M tokens\n\nOutput:\n\n$2.000 / 1M tokens\n\nGPT-5 nano\nThe fastest, cheapest version of GPT-5—great for summarization and\n\nclassification tasks\n\nPrice\n\nInput:\n\n$0.050 / 1M tokens\n\nCached input:\n\n$0.005 / 1M tokens\n\nOutput:\n\n$0.400 / 1M tokens\n\n8/21/25, 8:18 PM Pricing | OpenAI\n\nhttps://openai.com/api/pricing/ 2/11\n\nhttps://openai.com/\n\n\nGPT-4.1\n\nFine-tuning price\n\nInput:\n\n$3.00 / 1M tokens\n\nCached input:\n\n$0.75 / 1M tokens\n\nOutput:\n\n$12.00 / 1M tokens\n\nTraining:\n\n$25.00 / 1M tokens\n\nGPT-4.1 mini\n\nFine-tuning price\n\nInput:\n\n$0.80 / 1M tokens\n\nCached input:\n\n$0.20 / 1M tokens\n\nOutput:\n\n$3.20 / 1M tokens\n\nTraining:\n\n$5.00 / 1M tokens\n\nFine-tuning our models\n\nCustomize our models to get even higher performance for your specific use cases.\n\nAsk ChatGPT\n\n8/21/25, 8:18 PM Pricing | OpenAI\n\nhttps://openai.com/api/pricing/ 3/11\n\nhttps://openai.com/\n\n\nGPT-4.1 nano\n\nFine-tuning price\n\nInput:\n\n$0.20 / 1M tokens\n\nCached input:\n\n$0.05 / 1M tokens\n\nOutput:\n\n$0.80 / 1M tokens\n\nTraining:\n\n$1.50 / 1M tokens\n\no4-mini\n\nReinforcement fine-tuning price\n\nInput:\n\n$4.00 / 1M tokens\n\nCached input:\n\n$1.00 / 1M tokens\n\nOutput:\n\n$16.00 / 1M tokens\n\nTraining:\n\n$100.00 / training hour\n\nExplore detailed pricing\n\nOur APIs\n\n8/21/25, 8:18 PM Pricing | OpenAI\n\nhttps://openai.com/api/pricing/ 4/11",
                    ),
                ],
            ),
            ResponseOutputMessage(
                id="msg_68b0ef3a6ad8819299636c6596a724270c6a831e0341a521",
                content=[
                    ResponseOutputText(
                        annotations=[
                            AnnotationFileCitation(
                                file_id="file-K6W51Znon7LxKCv8B6AgoE",
                                filename="openai-pricing.pdf",
                                index=889,
                                type="file_citation",
                            )
                        ],
                        text="The cost of using GPT-5 via the OpenAI API is priced per million tokens as follows:\n\n- GPT-5 (the flagship model):\n  - Input tokens: $1.25 per 1M tokens\n  - Cached input tokens: $0.125 per 1M tokens\n  - Output tokens: $10.00 per 1M tokens\n\n- GPT-5 mini (a faster, cheaper version for well-defined tasks):\n  - Input tokens: $0.25 per 1M tokens\n  - Cached input tokens: $0.025 per 1M tokens\n  - Output tokens: $2.00 per 1M tokens\n\n- GPT-5 nano (the fastest, cheapest version for summarization and classification):\n  - Input tokens: $0.05 per 1M tokens\n  - Cached input tokens: $0.005 per 1M tokens\n  - Output tokens: $0.40 per 1M tokens\n\nThese prices reflect the API usage costs for processing tokens with the respective GPT-5 models. For more detailed pricing or enterprise options, contacting sales is recommended.\n\nThis pricing information is from the OpenAI pricing document you provided.",
                        type="output_text",
                        logprobs=[],
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            ),
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[
            FileSearchTool(
                type="file_search",
                vector_store_ids=["vs_68b0ec0ed2688191b8af10e2f14efc82"],
                filters=None,
                max_num_results=20,
                ranking_options=RankingOptions(ranker="auto", score_threshold=0.0),
            )
        ],
    )

    expected = """System: You are an AI Assistant that can call provided tools (a.k.a. functions). The set of available tools is provided to you as function signatures within <tools> </tools> XML tags. You may call one or more of these functions to assist with the user query. If the provided functions are not helpful/relevant, then just respond in natural conversational language. After you choose to call a function, you will be provided with the function's results within <tool_response> </tool_response> XML tags.

<tools>
{"type":"function","name":"file_search","description":"Search user-uploaded documents for relevant passages.","parameters":{"type":"object","properties":{"queries":{"type":"array","items":{"type":"string"},"description":"Search queries to run against the document index."}},"required":["queries"]}}
</tools>

For each function call return a JSON object, with the following pydantic model json schema:
{'name': <function-name>, 'arguments': <args-dict>}
Each function call should be enclosed within <tool_call> </tool_call> XML tags.
Example:
<tool_call>
{'name': <function-name>, 'arguments': <args-dict>}
</tool_call>
Note: Function calls and their results may optionally include a call_id, which should be ignored.

User: How much gpt-5 cost?

Assistant: <tool_call>
{
  "name": "file_search",
  "arguments": {
    "queries": [
      "cost of GPT-5",
      "price of GPT-5",
      "GPT-5 cost"
    ]
  },
  "call_id": "fs_68b0ef38cc0081928e29c0a0a29f8bb30c6a831e0341a521"
}
</tool_call>

Tool: <tool_response>
{
  "name": "file_search",
  "call_id": "fs_68b0ef38cc0081928e29c0a0a29f8bb30c6a831e0341a521",
  "output": [
    {
      "attributes": {},
      "file_id": "file-K6W51Znon7LxKCv8B6AgoE",
      "filename": "openai-pricing.pdf",
      "score": 0.9195,
      "text": "Pricing | OpenAI\\n\\n\\nPricing below reflects standard processing rates. To optimize cost and performance for\\n\\ndifferent use cases, we also offer:\\n\\nBatch API  : Save 50% on inputs and outputs with the Batch API and run tasks\\n\\nasynchronously over 24 hours.\\n\\nPriority processing  : offers reliable, high-speed performance with the flexibility to\\n\\npay-as-you-go.\\n\\nGPT-5\\n\\nThe best model for coding and agentic tasks across industries\\n\\nPrice\\n\\nAPI Pricing\\n\\nContact sales\\n\\nFlagship models\\n\\nOur frontier models designed to spend more time thinking before\\n\\nproducing a response, making them ideal for complex, multi-step problems.\\n\\n8/21/25, 8:18 PM Pricing | OpenAI\\n\\nhttps://openai.com/api/pricing/ 1/11\\n\\nhttps://platform.openai.com/docs/guides/batch\\nhttps://openai.com/api-priority-processing/\\nhttps://openai.com/contact-sales/\\nhttps://openai.com/\\n\\n\\nInput:\\n\\n$1.250 / 1M tokens\\n\\nCached input:\\n\\n$0.125 / 1M tokens\\n\\nOutput:\\n\\n$10.000 / 1M tokens\\n\\nGPT-5 mini\\nA faster, cheaper version of GPT-5 for well-defined tasks\\n\\nPrice\\n\\nInput:\\n\\n$0.250 / 1M tokens\\n\\nCached input:\\n\\n$0.025 / 1M tokens\\n\\nOutput:\\n\\n$2.000 / 1M tokens\\n\\nGPT-5 nano\\nThe fastest, cheapest version of GPT-5\\u2014great for summarization and\\n\\nclassification tasks\\n\\nPrice\\n\\nInput:\\n\\n$0.050 / 1M tokens\\n\\nCached input:\\n\\n$0.005 / 1M tokens\\n\\nOutput:\\n\\n$0.400 / 1M tokens\\n\\n8/21/25, 8:18 PM Pricing | OpenAI\\n\\nhttps://openai.com/api/pricing/ 2/11\\n\\nhttps://openai.com/\\n\\n\\nGPT-4.1\\n\\nFine-tuning price\\n\\nInput:\\n\\n$3.00 / 1M tokens\\n\\nCached input:\\n\\n$0.75 / 1M tokens\\n\\nOutput:\\n\\n$12.00 / 1M tokens\\n\\nTraining:\\n\\n$25.00 / 1M tokens\\n\\nGPT-4.1 mini\\n\\nFine-tuning price\\n\\nInput:\\n\\n$0.80 / 1M tokens\\n\\nCached input:\\n\\n$0.20 / 1M tokens\\n\\nOutput:\\n\\n$3.20 / 1M tokens\\n\\nTraining:\\n\\n$5.00 / 1M tokens\\n\\nFine-tuning our models\\n\\nCustomize our models to get even higher performance for your specific use cases.\\n\\nAsk ChatGPT\\n\\n8/21/25, 8:18 PM Pricing | OpenAI\\n\\nhttps://openai.com/api/pricing/ 3/11\\n\\nhttps://openai.com/\\n\\n\\nGPT-4.1 nano\\n\\nFine-tuning price\\n\\nInput:\\n\\n$0.20 / 1M tokens\\n\\nCached input:\\n\\n$0.05 / 1M tokens\\n\\nOutput:\\n\\n$0.80 / 1M tokens\\n\\nTraining:\\n\\n$1.50 / 1M tokens\\n\\no4-mini\\n\\nReinforcement fine-tuning price\\n\\nInput:\\n\\n$4.00 / 1M tokens\\n\\nCached input:\\n\\n$1.00 / 1M tokens\\n\\nOutput:\\n\\n$16.00 / 1M tokens\\n\\nTraining:\\n\\n$100.00 / training hour\\n\\nExplore detailed pricing\\n\\nOur APIs\\n\\n8/21/25, 8:18 PM Pricing | OpenAI\\n\\nhttps://openai.com/api/pricing/ 4/11"
    }
  ]
}
</tool_response>

Assistant:"""

    assert expected == form_prompt_string(
        [{"role": "user", "content": "How much gpt-5 cost?"}],
        response=response,
        use_responses=None,
        **openai_kwargs,  # type: ignore
    )


def test_form_prompt_string_responses_web_search() -> None:
    """Test form prompt string in OpenAI Responses with File Search."""
    openai_kwargs = {
        "tools": [{"type": "web_search"}],
        "tool_choice": {"type": "web_search"},
        "model": "gpt-4.1-mini",
        "input": "Give me a positive news story from today",
    }

    response = Response(
        id="resp_68b0fd14d644819294b3812274c4cda90ff59b7b98b47999",
        created_at=1756429588.0,
        model="gpt-4.1-mini-2025-04-14",
        object="response",
        output=[
            ResponseFunctionWebSearch(
                id="ws_68b0fd14ec4081929194473de6e8fc640ff59b7b98b47999",
                action=ActionSearch(
                    query="Give me a positive news story from today",
                    type="search",
                    sources=[
                        ActionSearchSource(
                            type="url",
                            url="https://www.podego.com/insights/august-2025-good-news-ai-pfas-stories?utm_source=openai",
                        ),
                    ],
                ),
                status="completed",
                type="web_search_call",
            ),
            ResponseOutputMessage(
                id="msg_68b0fd169f3081929bac951696c121250ff59b7b98b47999",
                content=[
                    ResponseOutputText(
                        annotations=[
                            AnnotationURLCitation(
                                end_index=892,
                                start_index=789,
                                title="Positive News Highlights | AI, PFAS Breakthroughs & More — August 2025 — Podego",
                                type="url_citation",
                                url="https://www.podego.com/insights/august-2025-good-news-ai-pfas-stories?utm_source=openai",
                            ),
                            AnnotationURLCitation(
                                end_index=1113,
                                start_index=943,
                                title="Positive News Highlights | AI, PFAS Breakthroughs & More — August 2025 — Podego",
                                type="url_citation",
                                url="https://www.podego.com/insights/august-2025-good-news-ai-pfas-stories?utm_source=openai",
                            ),
                        ],
                        text='Certainly! Here\'s a positive news story from today:\n\n**Dutch Cities Install Cat-Friendly Staircases Along Canals**\n\nIn Amsterdam and Amersfoort, Netherlands, authorities are implementing a simple yet effective solution to protect cats from drowning in canals. They are installing small staircases along the canal walls, allowing felines that fall in to climb out safely. Funded by a €100,000 environmental grant, the initiative aims to create over 500 cat-friendly escape routes by the end of the year. Judith Krom of the Party for the Animals emphasized the significance of this measure, stating, "A simple measure can prevent enormous animal suffering." This project highlights a compassionate approach to urban planning, ensuring the safety of non-human residents in city environments. ([podego.com](https://www.podego.com/insights/august-2025-good-news-ai-pfas-stories?utm_source=openai))\n\n\n## Positive News Highlights from August 2025:\n- [Positive News Highlights | AI, PFAS Breakthroughs & More — August 2025 — Podego](https://www.podego.com/insights/august-2025-good-news-ai-pfas-stories?utm_source=openai) ',
                        type="output_text",
                        logprobs=[],
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            ),
        ],
        parallel_tool_calls=True,
        temperature=0.0,
        tool_choice=ToolChoiceTypes(type="web_search_preview"),
        tools=[
            WebSearchTool(
                type="web_search",
                search_context_size="medium",
                user_location=UserLocation(
                    type="approximate",
                    city=None,
                    country="US",
                    region=None,
                    timezone=None,
                ),
            )
        ],
    )

    expected = """System: You are an AI Assistant that can call provided tools (a.k.a. functions). The set of available tools is provided to you as function signatures within <tools> </tools> XML tags. You may call one or more of these functions to assist with the user query. If the provided functions are not helpful/relevant, then just respond in natural conversational language. After you choose to call a function, you will be provided with the function's results within <tool_response> </tool_response> XML tags.

<tools>
{"type":"function","name":"web_search_call","description":"Search the web for relevant information.","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Search the web with a query and return relevant pages."}},"required":["query"]}}
</tools>

For each function call return a JSON object, with the following pydantic model json schema:
{'name': <function-name>, 'arguments': <args-dict>}
Each function call should be enclosed within <tool_call> </tool_call> XML tags.
Example:
<tool_call>
{'name': <function-name>, 'arguments': <args-dict>}
</tool_call>
Note: Function calls and their results may optionally include a call_id, which should be ignored.

User: Give me a positive news story from today

Assistant: <tool_call>
{
  "name": "web_search_call",
  "arguments": {
    "query": "Give me a positive news story from today"
  },
  "call_id": "ws_68b0fd14ec4081929194473de6e8fc640ff59b7b98b47999"
}
</tool_call>

Tool: <tool_response>
{
  "name": "web_search_call",
  "call_id": "ws_68b0fd14ec4081929194473de6e8fc640ff59b7b98b47999",
  "output": [
    {
      "url": "https://www.podego.com/insights/august-2025-good-news-ai-pfas-stories?utm_source=openai",
      "content": "MOCK CONTENT"
    }
  ]
}
</tool_response>

Assistant:"""

    from unittest.mock import patch

    with patch("trafilatura.extract", return_value="MOCK CONTENT"):
        assert expected == form_prompt_string(
            [{"role": "user", "content": "Give me a positive news story from today"}],
            response=response,
            use_responses=None,
            **openai_kwargs,  # type: ignore
        )


############### _FORM_PROMPT_CHAT_COMPLETIONS_API #######################
# Tests for the _form_prompt_chat_completions_api function.
#########################################################################


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


def test_form_prompt_chat_completions_api_single_user_no_tools_returns_content() -> None:
    messages = [
        {"role": "user", "content": "Say hi"},
    ]

    result = _form_prompt_chat_completions_api(cast(list["ChatCompletionMessageParam"], messages), tools=None)
    assert result == "Say hi"


def test_form_prompt_chat_completions_api_inserts_tools_after_system_block() -> None:
    # Two system messages, the tools prompt must be inserted after them
    messages = [
        {"role": "system", "content": "S1"},
        {"role": "system", "content": "S2"},
        {"role": "user", "content": "Ask"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "foo",
                "description": "Foo",
                "parameters": {"type": "object", "properties": {"x": {"type": "number"}}},
            },
        }
    ]

    prompt = _form_prompt_chat_completions_api(cast(list["ChatCompletionMessageParam"], messages), tools=tools)

    # Ensure order: System S1 -> System S2 -> tools prompt (contains <tools> and function name) -> User
    idx_s1 = prompt.find("System: S1\n\n")
    idx_s2 = prompt.find("S2\n\n")
    idx_tools = prompt.find("<tools>")
    idx_name = prompt.find('"name":"foo"')
    idx_user = prompt.find("User: Ask\n\n")
    assert -1 not in {idx_s1, idx_s2, idx_tools, idx_name, idx_user}
    assert idx_s1 < idx_s2 < idx_tools < idx_name < idx_user


def test_form_prompt_chat_completions_api_empty_tools_no_insertion() -> None:
    # Empty tools should behave as if no tools provided
    messages = [
        {"role": "user", "content": "Hello"},
    ]
    prompt = _form_prompt_chat_completions_api(cast(list["ChatCompletionMessageParam"], messages), tools=[])
    assert prompt == "Hello"


def test_form_prompt_chat_completions_api_with_tools_includes_tools_system_prompt() -> None:
    messages = [
        {"role": "user", "content": "Compute sum of 2 and 3"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add_numbers",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
            },
        }
    ]

    prompt = _form_prompt_chat_completions_api(cast(list["ChatCompletionMessageParam"], messages), tools=tools)

    expected = (
        "System: You are an AI Assistant that can call provided tools (a.k.a. functions). "
        "The set of available tools is provided to you as function signatures within <tools> </tools> XML tags. "
        "You may call one or more of these functions to assist with the user query. If the provided functions are not helpful/relevant, "
        "then just respond in natural conversational language. "
        "After you choose to call a function, you will be provided with the function's results within "
        "<tool_response> </tool_response> XML tags.\n\n"
        "<tools>\n"
        '{"type":"function","function":{"name":"add_numbers","description":"Add two numbers","parameters":{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}}}'
        "\n"
        "</tools>\n\n"
        "For each function call return a JSON object, with the following pydantic model json schema:\n"
        "{'name': <function-name>, 'arguments': <args-dict>}\n"
        "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
        "Example:\n"
        "<tool_call>\n"
        "{'name': <function-name>, 'arguments': <args-dict>}\n"
        "</tool_call>\n"
        "Note: Function calls and their results may optionally include a call_id, which should be ignored.\n\n"
        "User: Compute sum of 2 and 3\n\n"
        "Assistant:"
    )

    assert prompt == expected


def test_form_prompt_chat_completions_api_formats_tool_call_and_response() -> None:
    messages = [
        {"role": "user", "content": "What is weather?"},
        {
            "role": "assistant",
            "content": "I'll call the tool.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [{"temperature_c": 18}],
        },
    ]

    expected = (
        "User: What is weather?\n\n"
        "Assistant: I'll call the tool.\n\n"
        "<tool_call>\n"
        "{\n"
        '  "name": "get_weather",\n'
        '  "arguments": {\n'
        '    "location": "Paris"\n'
        "  },\n"
        '  "call_id": "call_1"\n'
        "}\n"
        "</tool_call>\n\n"
        "Tool: <tool_response>\n"
        "{\n"
        '  "name": "get_weather",\n'
        '  "call_id": "call_1",\n'
        '  "output": [\n'
        "    {\n"
        '      "temperature_c": 18\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "</tool_response>\n\n"
        "Assistant:"
    )

    result = _form_prompt_chat_completions_api(cast(list["ChatCompletionMessageParam"], messages))
    assert result == expected


@pytest.mark.filterwarnings("ignore:The last message is a tool call or assistant message")
def test_form_prompt_chat_completions_api_warns_when_last_is_assistant() -> None:
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Thinking..."},
    ]
    with pytest.warns(UserWarning, match="The last message is a tool call or assistant message"):
        _form_prompt_chat_completions_api(cast(list["ChatCompletionMessageParam"], messages))


@pytest.mark.filterwarnings("ignore:The last message is a tool call or assistant message")
def test_form_prompt_chat_completions_api_warns_when_last_has_tool_calls() -> None:
    messages = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        },
    ]
    with pytest.warns(UserWarning, match="The last message is a tool call or assistant message"):
        _form_prompt_chat_completions_api(cast(list["ChatCompletionMessageParam"], messages))


####################### _FORM_PROMPT_RESPONSES_API ######################
# Tests for the _form_prompt_responses_api function.
#########################################################################


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


########## FORM_RESPONSE_STRING_CHAT_COMPLETIONS_API ####################
# Tests for the form_response_string_chat_completions_api function.
#########################################################################


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


def test_form_response_string_chat_completions_api_empty_tool_calls_returns_content_only() -> None:
    response = {
        "content": "No actions needed.",
        "tool_calls": [],
    }
    # With empty tool_calls, it should just return the content
    assert form_response_string_chat_completions_api(response) == "No actions needed."


def test_form_response_string_chat_completions_api_none_content_with_tool_calls() -> None:
    response = {
        "content": None,
        "tool_calls": [
            {
                "function": {
                    "name": "sum",
                    "arguments": '{"a": 1, "b": 2}',
                }
            }
        ],
    }
    expected = (
        "<tool_call>\n"
        "{\n"
        '  "name": "sum",\n'
        '  "arguments": {\n'
        '    "a": 1,\n'
        '    "b": 2\n'
        "  }\n"
        "}\n"
        "</tool_call>"
    )
    assert form_response_string_chat_completions_api(response) == expected


@pytest.mark.filterwarnings("always:Error formatting tool_calls in response")
def test_form_response_string_chat_completions_api_arguments_is_dict_warns_and_returns_content() -> None:
    response = {
        "content": "Fallback to content",
        "tool_calls": [
            {
                "function": {
                    "name": "do_it",
                    # Incorrect type: should be a JSON string
                    "arguments": {"x": 1},
                }
            }
        ],
    }
    with pytest.warns(UserWarning, match="Error formatting tool_calls in response.*Returning content only"):
        assert form_response_string_chat_completions_api(response) == "Fallback to content"


@pytest.mark.filterwarnings("always:Error formatting tool_calls in response")
def test_form_response_string_chat_completions_api_tool_calls_not_list_warns_and_returns_content() -> None:
    response = {
        "content": "Only content",
        # Incorrect container type for tool_calls
        "tool_calls": {"function": {"name": "x", "arguments": "{}"}},
    }
    with pytest.warns(UserWarning, match="Error formatting tool_calls in response.*Returning content only"):
        assert form_response_string_chat_completions_api(response) == "Only content"


@pytest.mark.filterwarnings("always:Error formatting tool_calls in response")
def test_form_response_string_chat_completions_api_missing_function_fields_warns_and_returns_content() -> None:
    response = {
        "content": "Text",
        "tool_calls": [
            {
                # function key present but missing name field
                "function": {
                    "arguments": "{}",
                }
            }
        ],
    }
    with pytest.warns(UserWarning, match="Error formatting tool_calls in response: 'name'.*Returning content only"):
        assert form_response_string_chat_completions_api(response) == "Text"


@pytest.mark.filterwarnings("always:Error formatting tool_calls in response")
def test_form_response_string_chat_completions_api_whitespace_arguments_warns_and_returns_content() -> None:
    response = {
        "content": "Try again",
        "tool_calls": [
            {
                "function": {
                    "name": "noop",
                    "arguments": "  \n  ",
                }
            }
        ],
    }
    with pytest.warns(UserWarning, match="Error formatting tool_calls in response.*Returning content only"):
        assert form_response_string_chat_completions_api(response) == "Try again"


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
    with pytest.raises(TypeError, match="Expected response to be a dict or ChatCompletionMessage object, got str"):
        form_response_string_chat_completions_api("not a dict")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Expected response to be a dict or ChatCompletionMessage object, got list"):
        form_response_string_chat_completions_api([])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Expected response to be a dict or ChatCompletionMessage object, got NoneType"):
        form_response_string_chat_completions_api(None)  # type: ignore[arg-type]


@pytest.mark.filterwarnings("always:Error formatting tool_calls in response")
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


def test_form_response_string_chat_completions_api_chatcompletion_message_just_content() -> None:
    """Test form_response_string_chat_completions_api with ChatCompletionMessage containing just content."""

    content = "Hello, how can I help you today?"
    message = ChatCompletionMessage(
        role="assistant",
        content=content,
    )
    result = form_response_string_chat_completions_api(message)
    assert result == content


def test_form_response_string_chat_completions_api_chatcompletion_message_just_tool_calls() -> None:
    """Test form_response_string_chat_completions_api with ChatCompletionMessage containing just tool calls."""
    message = ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call_123",
                function=Function(
                    name="search_restaurants",
                    arguments='{"city": "Tokyo", "cuisine_type": "sushi", "max_price": 150, "dietary_restrictions": ["vegetarian", "gluten-free"], "open_now": true}',
                ),
                type="function",
            )
        ],
    )
    expected = (
        "<tool_call>\n"
        "{\n"
        '  "name": "search_restaurants",\n'
        '  "arguments": {\n'
        '    "city": "Tokyo",\n'
        '    "cuisine_type": "sushi",\n'
        '    "max_price": 150,\n'
        '    "dietary_restrictions": [\n'
        '      "vegetarian",\n'
        '      "gluten-free"\n'
        "    ],\n"
        '    "open_now": true\n'
        "  }\n"
        "}\n"
        "</tool_call>"
    )
    result = form_response_string_chat_completions_api(message)
    assert result == expected


def test_form_response_string_chat_completions_api_chatcompletion_message_content_and_tool_calls() -> None:
    """Test form_response_string_chat_completions_api with ChatCompletionMessage containing both content and tool calls."""
    message = ChatCompletionMessage(
        role="assistant",
        content="I'll check the weather for you.",
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call_123",
                function=Function(
                    name="get_weather",
                    arguments='{"location": "Paris"}',
                ),
                type="function",
            )
        ],
    )
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
    result = form_response_string_chat_completions_api(message)
    assert result == expected


def test_form_response_string_chat_completions_api_chatcompletion_message_multiple_tool_calls() -> None:
    """Test form_response_string_chat_completions_api with ChatCompletionMessage containing multiple tool calls."""
    message = ChatCompletionMessage(
        role="assistant",
        content="Let me check multiple things for you.",
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call_123",
                function=Function(
                    name="get_weather",
                    arguments='{"location": "Paris"}',
                ),
                type="function",
            ),
            ChatCompletionMessageToolCall(
                id="call_456",
                function=Function(
                    name="get_time",
                    arguments='{"timezone": "UTC"}',
                ),
                type="function",
            ),
        ],
    )
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
    result = form_response_string_chat_completions_api(message)
    assert result == expected


def test_form_response_string_chat_completions_api_chatcompletion_message_empty_content() -> None:
    """Test form_response_string_chat_completions_api with ChatCompletionMessage containing empty content."""
    message = ChatCompletionMessage(
        role="assistant",
        content="",
    )
    expected = ""
    result = form_response_string_chat_completions_api(message)
    assert result == expected


def test_form_response_string_chat_completions_api_chatcompletion_message_empty_arguments() -> None:
    """Test form_response_string_chat_completions_api with ChatCompletionMessage containing empty arguments."""
    message = ChatCompletionMessage(
        role="assistant",
        content="Running action",
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call_123",
                function=Function(
                    name="execute_action",
                    arguments="",
                ),
                type="function",
            )
        ],
    )
    expected = (
        "Running action\n"
        "<tool_call>\n"
        "{\n"
        '  "name": "execute_action",\n'
        '  "arguments": {}\n'
        "}\n"
        "</tool_call>"
    )
    result = form_response_string_chat_completions_api(message)
    assert result == expected


def test_form_response_string_chat_completions_api_chatcompletion_message_none_content() -> None:
    """Test form_response_string_chat_completions_api with ChatCompletionMessage containing None content."""
    message = ChatCompletionMessage(
        role="assistant",
        content=None,
    )
    expected = ""
    result = form_response_string_chat_completions_api(message)
    assert result == expected


def test_form_response_string_chat_completions_just_content() -> None:
    """Test form_response_string_chat_completions with ChatCompletion containing just content."""

    content = "Hello, how can I help you today?"

    message = ChatCompletionMessage(role="assistant", content=content)
    response = ChatCompletion(
        id="test",
        choices=[
            Choice(
                index=0,
                message=message,
                finish_reason="stop",
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )

    result = form_response_string_chat_completions(response)
    assert result == content

    assert result == form_response_string_chat_completions_api(message)


def test_form_response_string_chat_completions_multiple_choices() -> None:
    """Test form_response_string_chat_completions with ChatCompletion containing multiple choices."""

    content_first = "Hello, how can I help you today?"
    content_second = "Hi there! What can I do for you?"

    message_first = ChatCompletionMessage(role="assistant", content=content_first)
    message_second = ChatCompletionMessage(role="assistant", content=content_second)
    response = ChatCompletion(
        id="test",
        choices=[
            Choice(
                index=0,
                message=message_first,
                finish_reason="stop",
            ),
            Choice(
                index=1,
                message=message_second,
                finish_reason="stop",
            ),
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )

    result = form_response_string_chat_completions(response)
    assert result == content_first

    assert result == form_response_string_chat_completions_api(message_first)


def test_form_response_string_chat_completions_uses_api_function() -> None:
    """Test that form_response_string_chat_completions calls form_response_string_chat_completions_api."""
    from unittest.mock import patch

    message = ChatCompletionMessage(role="assistant", content="Test response")
    response = ChatCompletion(
        id="test",
        choices=[
            Choice(
                index=0,
                message=message,
                finish_reason="stop",
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )

    # Mock the api function and test that it's called
    with patch("cleanlab_tlm.utils.chat.form_response_string_chat_completions_api") as mock_api_func:
        mock_api_func.return_value = "Mocked response"

        result = form_response_string_chat_completions(response)

        mock_api_func.assert_called_once_with(message)
        assert result == "Mocked response"


############ FORM_RESPONSE_STRING_RESPONSES_API ##########################
# Tests for the form_response_string_responses_api function.
#########################################################################


def test_form_response_string_responses_file_search() -> None:
    """Test form prompt string in OpenAI Responses with File Search."""

    response = Response(
        id="resp_68b0ef38417481929bcf0b9cb8f884120c6a831e0341a521",
        created_at=1756426040.0,
        model="gpt-4.1-mini-2025-04-14",
        object="response",
        output=[
            ResponseFileSearchToolCall(
                id="fs_68b0ef38cc0081928e29c0a0a29f8bb30c6a831e0341a521",
                queries=["cost of GPT-5", "price of GPT-5", "GPT-5 cost"],
                status="completed",
                type="file_search_call",
                results=[
                    Result(
                        attributes={},
                        file_id="file-K6W51Znon7LxKCv8B6AgoE",
                        filename="openai-pricing.pdf",
                        score=0.9195,
                        text="Pricing | OpenAI\n\n\nPricing below reflects standard processing rates. To optimize cost and performance for\n\ndifferent use cases, we also offer:\n\nBatch API  : Save 50% on inputs and outputs with the Batch API and run tasks\n\nasynchronously over 24 hours.\n\nPriority processing  : offers reliable, high-speed performance with the flexibility to\n\npay-as-you-go.\n\nGPT-5\n\nThe best model for coding and agentic tasks across industries\n\nPrice\n\nAPI Pricing\n\nContact sales\n\nFlagship models\n\nOur frontier models designed to spend more time thinking before\n\nproducing a response, making them ideal for complex, multi-step problems.\n\n8/21/25, 8:18 PM Pricing | OpenAI\n\nhttps://openai.com/api/pricing/ 1/11\n\nhttps://platform.openai.com/docs/guides/batch\nhttps://openai.com/api-priority-processing/\nhttps://openai.com/contact-sales/\nhttps://openai.com/\n\n\nInput:\n\n$1.250 / 1M tokens\n\nCached input:\n\n$0.125 / 1M tokens\n\nOutput:\n\n$10.000 / 1M tokens\n\nGPT-5 mini\nA faster, cheaper version of GPT-5 for well-defined tasks\n\nPrice\n\nInput:\n\n$0.250 / 1M tokens\n\nCached input:\n\n$0.025 / 1M tokens\n\nOutput:\n\n$2.000 / 1M tokens\n\nGPT-5 nano\nThe fastest, cheapest version of GPT-5—great for summarization and\n\nclassification tasks\n\nPrice\n\nInput:\n\n$0.050 / 1M tokens\n\nCached input:\n\n$0.005 / 1M tokens\n\nOutput:\n\n$0.400 / 1M tokens\n\n8/21/25, 8:18 PM Pricing | OpenAI\n\nhttps://openai.com/api/pricing/ 2/11\n\nhttps://openai.com/\n\n\nGPT-4.1\n\nFine-tuning price\n\nInput:\n\n$3.00 / 1M tokens\n\nCached input:\n\n$0.75 / 1M tokens\n\nOutput:\n\n$12.00 / 1M tokens\n\nTraining:\n\n$25.00 / 1M tokens\n\nGPT-4.1 mini\n\nFine-tuning price\n\nInput:\n\n$0.80 / 1M tokens\n\nCached input:\n\n$0.20 / 1M tokens\n\nOutput:\n\n$3.20 / 1M tokens\n\nTraining:\n\n$5.00 / 1M tokens\n\nFine-tuning our models\n\nCustomize our models to get even higher performance for your specific use cases.\n\nAsk ChatGPT\n\n8/21/25, 8:18 PM Pricing | OpenAI\n\nhttps://openai.com/api/pricing/ 3/11\n\nhttps://openai.com/\n\n\nGPT-4.1 nano\n\nFine-tuning price\n\nInput:\n\n$0.20 / 1M tokens\n\nCached input:\n\n$0.05 / 1M tokens\n\nOutput:\n\n$0.80 / 1M tokens\n\nTraining:\n\n$1.50 / 1M tokens\n\no4-mini\n\nReinforcement fine-tuning price\n\nInput:\n\n$4.00 / 1M tokens\n\nCached input:\n\n$1.00 / 1M tokens\n\nOutput:\n\n$16.00 / 1M tokens\n\nTraining:\n\n$100.00 / training hour\n\nExplore detailed pricing\n\nOur APIs\n\n8/21/25, 8:18 PM Pricing | OpenAI\n\nhttps://openai.com/api/pricing/ 4/11",
                    )
                ],
            ),
            ResponseOutputMessage(
                id="msg_68b0ef3a6ad8819299636c6596a724270c6a831e0341a521",
                content=[
                    ResponseOutputText(
                        annotations=[
                            AnnotationFileCitation(
                                file_id="file-K6W51Znon7LxKCv8B6AgoE",
                                filename="openai-pricing.pdf",
                                index=889,
                                type="file_citation",
                            )
                        ],
                        text="The cost of using GPT-5 via the OpenAI API is priced per million tokens as follows:\n\n- GPT-5 (the flagship model):\n  - Input tokens: $1.25 per 1M tokens\n  - Cached input tokens: $0.125 per 1M tokens\n  - Output tokens: $10.00 per 1M tokens\n\n- GPT-5 mini (a faster, cheaper version for well-defined tasks):\n  - Input tokens: $0.25 per 1M tokens\n  - Cached input tokens: $0.025 per 1M tokens\n  - Output tokens: $2.00 per 1M tokens\n\n- GPT-5 nano (the fastest, cheapest version for summarization and classification):\n  - Input tokens: $0.05 per 1M tokens\n  - Cached input tokens: $0.005 per 1M tokens\n  - Output tokens: $0.40 per 1M tokens\n\nThese prices reflect the API usage costs for processing tokens with the respective GPT-5 models. For more detailed pricing or enterprise options, contacting sales is recommended.\n\nThis pricing information is from the OpenAI pricing document you provided.",
                        type="output_text",
                        logprobs=[],
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            ),
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[
            FileSearchTool(
                type="file_search",
                vector_store_ids=["vs_68b0ec0ed2688191b8af10e2f14efc82"],
                filters=None,
                max_num_results=20,
                ranking_options=RankingOptions(ranker="auto", score_threshold=0.0),
            )
        ],
    )

    expected = """The cost of using GPT-5 via the OpenAI API is priced per million tokens as follows:

- GPT-5 (the flagship model):
  - Input tokens: $1.25 per 1M tokens
  - Cached input tokens: $0.125 per 1M tokens
  - Output tokens: $10.00 per 1M tokens

- GPT-5 mini (a faster, cheaper version for well-defined tasks):
  - Input tokens: $0.25 per 1M tokens
  - Cached input tokens: $0.025 per 1M tokens
  - Output tokens: $2.00 per 1M tokens

- GPT-5 nano (the fastest, cheapest version for summarization and classification):
  - Input tokens: $0.05 per 1M tokens
  - Cached input tokens: $0.005 per 1M tokens
  - Output tokens: $0.40 per 1M tokens

These prices reflect the API usage costs for processing tokens with the respective GPT-5 models. For more detailed pricing or enterprise options, contacting sales is recommended.

This pricing information is from the OpenAI pricing document you provided."""

    assert expected == form_response_string_responses_api(
        response=response,
    )


def test_form_response_string_responses_web_search() -> None:
    """Test form prompt string in OpenAI Responses with File Search."""

    response = Response(
        id="resp_68b0fd14d644819294b3812274c4cda90ff59b7b98b47999",
        created_at=1756429588.0,
        model="gpt-4.1-mini-2025-04-14",
        object="response",
        output=[
            ResponseFunctionWebSearch(
                id="ws_68b0fd14ec4081929194473de6e8fc640ff59b7b98b47999",
                action=ActionSearch(
                    query="Give me a positive news story from today",
                    type="search",
                    sources=[
                        ActionSearchSource(
                            type="url",
                            url="https://www.podego.com/insights/august-2025-good-news-ai-pfas-stories?utm_source=openai",
                        ),
                    ],
                ),
                status="completed",
                type="web_search_call",
            ),
            ResponseOutputMessage(
                id="msg_68b0fd169f3081929bac951696c121250ff59b7b98b47999",
                content=[
                    ResponseOutputText(
                        annotations=[
                            AnnotationURLCitation(
                                end_index=892,
                                start_index=789,
                                title="Positive News Highlights | AI, PFAS Breakthroughs & More — August 2025 — Podego",
                                type="url_citation",
                                url="https://www.podego.com/insights/august-2025-good-news-ai-pfas-stories?utm_source=openai",
                            ),
                            AnnotationURLCitation(
                                end_index=1113,
                                start_index=943,
                                title="Positive News Highlights | AI, PFAS Breakthroughs & More — August 2025 — Podego",
                                type="url_citation",
                                url="https://www.podego.com/insights/august-2025-good-news-ai-pfas-stories?utm_source=openai",
                            ),
                        ],
                        text='Certainly! Here\'s a positive news story from today:\n\n**Dutch Cities Install Cat-Friendly Staircases Along Canals**\n\nIn Amsterdam and Amersfoort, Netherlands, authorities are implementing a simple yet effective solution to protect cats from drowning in canals. They are installing small staircases along the canal walls, allowing felines that fall in to climb out safely. Funded by a €100,000 environmental grant, the initiative aims to create over 500 cat-friendly escape routes by the end of the year. Judith Krom of the Party for the Animals emphasized the significance of this measure, stating, "A simple measure can prevent enormous animal suffering." This project highlights a compassionate approach to urban planning, ensuring the safety of non-human residents in city environments. ([podego.com](https://www.podego.com/insights/august-2025-good-news-ai-pfas-stories?utm_source=openai))\n\n\n## Positive News Highlights from August 2025:\n- [Positive News Highlights | AI, PFAS Breakthroughs & More — August 2025 — Podego](https://www.podego.com/insights/august-2025-good-news-ai-pfas-stories?utm_source=openai) ',
                        type="output_text",
                        logprobs=[],
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            ),
        ],
        parallel_tool_calls=True,
        temperature=0.0,
        tool_choice=ToolChoiceTypes(type="web_search_preview"),
        tools=[
            WebSearchTool(
                type="web_search",
                search_context_size="medium",
                user_location=UserLocation(
                    type="approximate",
                    city=None,
                    country="US",
                    region=None,
                    timezone=None,
                ),
            )
        ],
    )

    expected = """Certainly! Here's a positive news story from today:

**Dutch Cities Install Cat-Friendly Staircases Along Canals**

In Amsterdam and Amersfoort, Netherlands, authorities are implementing a simple yet effective solution to protect cats from drowning in canals. They are installing small staircases along the canal walls, allowing felines that fall in to climb out safely. Funded by a €100,000 environmental grant, the initiative aims to create over 500 cat-friendly escape routes by the end of the year. Judith Krom of the Party for the Animals emphasized the significance of this measure, stating, "A simple measure can prevent enormous animal suffering." This project highlights a compassionate approach to urban planning, ensuring the safety of non-human residents in city environments. ([podego.com](https://www.podego.com/insights/august-2025-good-news-ai-pfas-stories?utm_source=openai))


## Positive News Highlights from August 2025:
- [Positive News Highlights | AI, PFAS Breakthroughs & More — August 2025 — Podego](https://www.podego.com/insights/august-2025-good-news-ai-pfas-stories?utm_source=openai)"""

    from unittest.mock import patch

    with patch("trafilatura.extract", return_value="MOCK CONTENT"):
        assert expected == form_response_string_responses_api(
            response=response,
        )


###################### IS_TOOL_CALL_RESPONSE ##############################
# Tests for the _is_tool_call_response function.
#########################################################################


class TestIsToolCallResponse:
    """Test suite for the _is_tool_call_response function."""

    # pytest.param is required to create test cases with readable ids
    # the lambda function approach in the pytest.mark.parametrize(ids=...) is not working.
    @pytest.mark.parametrize(
        "response_text,expected",  # noqa: PT006
        [
            pytest.param(
                """<tool_call>
{
  "name": "get_weather",
  "arguments": {
    "location": "New York"
  }
}
</tool_call>""",
                True,
                id="basic_tool_call",
            ),
            pytest.param(
                """
    <tool_call>
{
  "name": "calculate",
  "arguments": {"x": 10, "y": 5}
}
</tool_call>   """,
                True,
                id="tool_call_with_whitespace",
            ),
            pytest.param(
                """<tool_call>
{"name": "function1", "arguments": {}}
</tool_call>
<tool_call>
{"name": "function2", "arguments": {}}
</tool_call>""",
                True,
                id="consecutive_tool_calls",
            ),
            pytest.param("This is a regular text response from the assistant.", False, id="regular_text"),
            pytest.param("", False, id="empty_string"),
            pytest.param("None", False, id="none_as_string"),
            pytest.param("<tool_cal", False, id="partial_tag"),
            pytest.param(
                """<incorrect_tag>
{"name": "function", "arguments": {}}
</incorrect_tag>""",
                False,
                id="incorrect_tag",
            ),
            pytest.param(
                """Here is some text before the tool call.
<tool_call>
{"name": "function", "arguments": {}}
</tool_call>""",
                False,
                id="text_before_tool_call",
            ),
            pytest.param(
                """<tool_call>
{"name": "function", "arguments": {}}
</tool_call>
And here is some text after the tool call.""",
                False,
                id="text_after_tool_call",
            ),
            pytest.param(
                """
<tool_call>
{"name": "first_function", "arguments": {"param": "value"}}
</tool_call>
Here is some explanatory text between two tool calls.
<tool_call>
{"name": "second_function", "arguments": {"other_param": 42}}
</tool_call>""",
                False,
                id="text_between_tool_calls",
            ),
            pytest.param(
                """Starting with some text.
<tool_call>
{"name": "function1", "arguments": {}}
</tool_call>
Text in the middle.
<tool_call>
{"name": "function2", "arguments": {}}
</tool_call>
Ending with more text.""",
                False,
                id="text_everywhere",
            ),
        ],
    )
    def test_is_tool_call_response(self, response_text: str, expected: bool) -> None:
        """Test _is_tool_call_response with various input scenarios."""
        assert _is_tool_call_response(response_text) is expected

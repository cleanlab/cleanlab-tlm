import pytest

from cleanlab_tlm.utils.chat import form_prompt_string


def test_form_prompt_string_multiple_messages() -> None:
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


def test_form_prompt_string_single_message() -> None:
    messages = [{"role": "user", "content": "Just one message."}]
    assert form_prompt_string(messages) == "Just one message."


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


def test_form_prompt_string_with_tools() -> None:
    messages = [
        {"role": "user", "content": "What can you do?"},
    ]
    tools = [
        {
            "name": "search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query"}},
                "required": ["query"],
            },
        }
    ]
    expected = (
        "System: You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. "
        "You may call one or more functions to assist with the user query. If available tools are not relevant in assisting "
        "with user query, just respond in natural conversational language. Don't make assumptions about what values to plug "
        "into functions. After calling & executing the functions, you will be provided with function results within "
        "<tool_response> </tool_response> XML tags.\n\n"
        "<tools>\n"
        '{"type":"function","function":{"name":"search","description":"Search the web for information","parameters":'
        '{"type":"object","properties":{"query":{"type":"string","description":"The search query"}},"required":["query"]}}}\n'
        "</tools>\n\n"
        "For each function call return a JSON object, with the following pydantic model json schema:\n"
        "{'title': 'FunctionCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, "
        "'arguments': {'title': 'Arguments', 'type': 'object'}}, 'required': ['arguments', 'name']}\n"
        "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
        "Example:\n"
        "<tool_call>\n"
        "{'name': <function-name>, 'arguments': <args-dict>}\n"
        "</tool_call>\n\n"
        "User: What can you do?\n\n"
        "Assistant:"
    )
    assert form_prompt_string(messages, tools) == expected


def test_form_prompt_string_with_tools_and_system() -> None:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What can you do?"},
    ]
    tools = [
        {
            "name": "calculator",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "The mathematical expression"}},
                "required": ["expression"],
            },
        }
    ]
    expected = (
        "System: You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. "
        "You may call one or more functions to assist with the user query. If available tools are not relevant in assisting "
        "with user query, just respond in natural conversational language. Don't make assumptions about what values to plug "
        "into functions. After calling & executing the functions, you will be provided with function results within "
        "<tool_response> </tool_response> XML tags.\n\n"
        "<tools>\n"
        '{"type":"function","function":{"name":"calculator","description":"Perform mathematical calculations","parameters":'
        '{"type":"object","properties":{"expression":{"type":"string","description":"The mathematical expression"}},"required":["expression"]}}}\n'
        "</tools>\n\n"
        "For each function call return a JSON object, with the following pydantic model json schema:\n"
        "{'title': 'FunctionCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, "
        "'arguments': {'title': 'Arguments', 'type': 'object'}}, 'required': ['arguments', 'name']}\n"
        "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
        "Example:\n"
        "<tool_call>\n"
        "{'name': <function-name>, 'arguments': <args-dict>}\n"
        "</tool_call>\n\n"
        "System: You are a helpful assistant.\n\n"
        "User: What can you do?\n\n"
        "Assistant:"
    )
    assert form_prompt_string(messages, tools) == expected


def test_form_prompt_string_warns_on_tool_call_last() -> None:
    """Test that a warning is raised when the last message is a tool call."""
    messages = [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "type": "function_call",
            "id": "fc_12345xyz",
            "call_id": "call_12345xyz",
            "name": "get_weather",
            "arguments": '{"latitude":48.8566,"longitude":2.3522}',
        },
    ]
    expected = (
        "User: What's the weather in Paris?\n\n"
        "Assistant: <tool_call>\n"
        "{\n"
        '  "name": "get_weather",\n'
        '  "arguments": {\n'
        '    "latitude": 48.8566,\n'
        '    "longitude": 2.3522\n'
        "  }\n"
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


def test_form_prompt_string_with_function_call_output() -> None:
    messages = [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "type": "function_call",
            "id": "fc_12345xyz",
            "call_id": "call_12345xyz",
            "name": "get_weather",
            "arguments": '{"latitude":48.8566,"longitude":2.3522}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_12345xyz",
            "output": "22.1",
        },
    ]
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather for a location using latitude and longitude",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "The latitude coordinate"},
                    "longitude": {"type": "number", "description": "The longitude coordinate"},
                },
                "required": ["latitude", "longitude"],
            },
        }
    ]
    expected = (
        "System: You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. "
        "You may call one or more functions to assist with the user query. If available tools are not relevant in assisting "
        "with user query, just respond in natural conversational language. Don't make assumptions about what values to plug "
        "into functions. After calling & executing the functions, you will be provided with function results within "
        "<tool_response> </tool_response> XML tags.\n\n"
        "<tools>\n"
        '{"type":"function","function":{"name":"get_weather","description":"Get weather for a location using latitude and longitude",'
        '"parameters":{"type":"object","properties":{"latitude":{"type":"number","description":"The latitude coordinate"},'
        '"longitude":{"type":"number","description":"The longitude coordinate"}},"required":["latitude","longitude"]}}}\n'
        "</tools>\n\n"
        "For each function call return a JSON object, with the following pydantic model json schema:\n"
        "{'title': 'FunctionCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, "
        "'arguments': {'title': 'Arguments', 'type': 'object'}}, 'required': ['arguments', 'name']}\n"
        "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
        "Example:\n"
        "<tool_call>\n"
        "{'name': <function-name>, 'arguments': <args-dict>}\n"
        "</tool_call>\n\n"
        "User: What's the weather in Paris?\n\n"
        "Assistant: <tool_call>\n"
        "{\n"
        '  "name": "get_weather",\n'
        '  "arguments": {\n'
        '    "latitude": 48.8566,\n'
        '    "longitude": 2.3522\n'
        "  }\n"
        "}\n"
        "</tool_call>\n\n"
        "<tool_response>\n"
        "{\n"
        '  "name": "get_weather",\n'
        '  "call_id": "call_12345xyz",\n'
        '  "output": "22.1"\n'
        "}\n"
        "</tool_response>\n\n"
        "Assistant:"
    )
    assert form_prompt_string(messages, tools) == expected

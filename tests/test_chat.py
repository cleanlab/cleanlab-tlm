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
    expected = "You are a helpful assistant.\n\n" "User: What is the weather?\n\n" "Assistant:"
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

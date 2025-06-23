import pytest
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion import Choice

from cleanlab_tlm.internal.types import TLMQualityPreset
from cleanlab_tlm.utils.chat_completions import TLMChatCompletion
from tests.conftest import make_text_unique
from tests.constants import TEST_PROMPT, TEST_RESPONSE
from tests.test_get_trustworthiness_score import is_trustworthiness_score_json_format

test_prompt = make_text_unique(TEST_PROMPT)
test_response = make_text_unique(TEST_RESPONSE)


@pytest.mark.parametrize(
    "quality_preset",
    ["base", "low", "medium"],
)
def test_tlm_chat_completion_score(quality_preset: TLMQualityPreset) -> None:
    tlm_chat = TLMChatCompletion(quality_preset=quality_preset)
    openai_kwargs = {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": test_prompt}],
    }
    response = ChatCompletion(
        id="test",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=test_response),
                finish_reason="stop",
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )

    score = tlm_chat.score(response=response, **openai_kwargs)

    assert score is not None
    assert is_trustworthiness_score_json_format(score)


def test_tlm_chat_completion_score_with_options() -> None:
    tlm_chat = TLMChatCompletion(options={"log": ["explanation", "perplexity"]})
    openai_kwargs = {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": test_prompt}],
    }
    response = ChatCompletion(
        id="test",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=test_response),
                finish_reason="stop",
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )

    score = tlm_chat.score(response=response, **openai_kwargs)

    assert score is not None
    assert is_trustworthiness_score_json_format(score)


def test_tlm_chat_completion_score_with_tools() -> None:
    tlm_chat = TLMChatCompletion()
    openai_kwargs = {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": test_prompt}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query",
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        ],
    }
    response = ChatCompletion(
        id="test",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=test_response),
                finish_reason="stop",
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )

    score = tlm_chat.score(response=response, **openai_kwargs)

    assert score is not None
    assert is_trustworthiness_score_json_format(score)


def test_tlm_chat_completion_score_invalid_response() -> None:
    tlm_chat = TLMChatCompletion()
    openai_kwargs = {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": test_prompt}],
    }
    invalid_response = {"invalid": "response"}

    with pytest.raises(TypeError, match="The response is not an OpenAI ChatCompletion object."):
        tlm_chat.score(response=invalid_response, **openai_kwargs)  # type: ignore


def test_tlm_chat_completion_score_missing_messages() -> None:
    tlm_chat = TLMChatCompletion()
    openai_kwargs = {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": test_prompt}],
    }
    response = ChatCompletion(
        id="test",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=None),
                finish_reason="stop",
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )

    with pytest.raises(
        ValueError,
        match="The OpenAI ChatCompletion object does not contain a message content.",
    ):
        tlm_chat.score(response=response, **openai_kwargs)

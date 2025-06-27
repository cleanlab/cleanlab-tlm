import pytest
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion import Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob

from cleanlab_tlm.internal.types import TLMQualityPreset
from cleanlab_tlm.utils.chat_completions import TLMChatCompletion, _extract_perplexity
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
    assert score["log"]["explanation"] is not None
    assert score["log"]["perplexity"] is None


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


def test_tlm_chat_completion_score_with_perplexity() -> None:
    tlm_chat = TLMChatCompletion(options={"log": ["perplexity"]})
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
                logprobs=ChoiceLogprobs(
                    content=[
                        ChatCompletionTokenLogprob(
                            token="The",  # noqa: S106
                            bytes=[84, 104, 101],
                            logprob=0.0,
                            top_logprobs=[],
                        ),
                        ChatCompletionTokenLogprob(
                            token=" capital",  # noqa: S106
                            bytes=[32, 99, 97, 112, 105, 116, 97, 108],
                            logprob=0.0,
                            top_logprobs=[],
                        ),
                        ChatCompletionTokenLogprob(
                            token=" of",  # noqa: S106
                            bytes=[32, 111, 102],
                            logprob=0.0,
                            top_logprobs=[],
                        ),
                        ChatCompletionTokenLogprob(
                            token=" France",  # noqa: S106
                            bytes=[32, 70, 114, 97, 110, 99, 101],
                            logprob=0.0,
                            top_logprobs=[],
                        ),
                        ChatCompletionTokenLogprob(
                            token=" is",  # noqa: S106
                            bytes=[32, 105, 115],
                            logprob=0.0,
                            top_logprobs=[],
                        ),
                        ChatCompletionTokenLogprob(
                            token=" Paris",  # noqa: S106
                            bytes=[32, 80, 97, 114, 105, 115],
                            logprob=0.0,
                            top_logprobs=[],
                        ),
                        ChatCompletionTokenLogprob(
                            token=".",  # noqa: S106
                            bytes=[46],
                            logprob=-1.9361264946837764e-07,
                            top_logprobs=[],
                        ),
                    ],
                    refusal=None,
                ),
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )

    manually_calculated_perplexity = _extract_perplexity(response)

    score = tlm_chat.score(response=response, **openai_kwargs)
    returned_perplexity = score["log"]["perplexity"]

    assert manually_calculated_perplexity == returned_perplexity


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

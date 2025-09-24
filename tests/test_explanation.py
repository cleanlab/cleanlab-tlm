from typing import cast

import pytest

from cleanlab_tlm.errors import ValidationError
from cleanlab_tlm.tlm import TLM, TLMResponse, TLMScore
from cleanlab_tlm.utils.chat_completions import TLMChatCompletion
from cleanlab_tlm.utils.rag import (
    EvalMetric,
    TrustworthyRAG,
    TrustworthyRAGResponse,
    TrustworthyRAGScore,
)
from tests.conftest import make_text_unique
from tests.constants import TEST_PROMPT, TEST_PROMPT_BATCH
from tests.test_tlm_rag import (
    TEST_CONTEXT,
    TEST_CONTEXT_BATCH,
    TEST_QUERY,
    TEST_QUERY_BATCH,
)

test_prompt = make_text_unique(TEST_PROMPT)
test_prompt_batch = [make_text_unique(prompt) for prompt in TEST_PROMPT_BATCH]
test_context = make_text_unique(TEST_CONTEXT)
test_context_batch = [make_text_unique(context) for context in TEST_CONTEXT_BATCH]
test_query = make_text_unique(TEST_QUERY)
test_query_batch = [make_text_unique(query) for query in TEST_QUERY_BATCH]

TEST_BAD_RESPONSE = "Washington DC"
TEST_BAD_RESPONSE_BATCH = ["Washington DC", "Paris"]


def test_get_explanation_single_tlm_prompt(tlm: TLM) -> None:
    tlm_result = TLMResponse(
        response=TEST_BAD_RESPONSE,
        trustworthiness_score=0.1,
    )

    explanation = tlm.get_explanation(
        prompt=test_prompt,
        tlm_result=tlm_result,
    )

    assert isinstance(explanation, str)
    assert len(explanation) > 0
    assert "log" in tlm_result
    assert "explanation" in tlm_result["log"]
    assert tlm_result["log"]["explanation"] == explanation


def test_get_explanation_single_tlm_score(tlm: TLM) -> None:
    tlm_result = TLMScore(trustworthiness_score=0.05)

    explanation = tlm.get_explanation(
        prompt=test_prompt,
        response=TEST_BAD_RESPONSE,
        tlm_result=tlm_result,
    )

    assert isinstance(explanation, str)
    assert len(explanation) > 0
    assert "log" in tlm_result
    assert "explanation" in tlm_result["log"]
    assert tlm_result["log"]["explanation"] == explanation


def test_get_explanation_batch_tlm_prompt(tlm: TLM) -> None:
    """Test getting explanations for a batch of TLMResponse objects."""

    tlm_results = [TLMResponse(response=response, trustworthiness_score=0.1) for response in TEST_BAD_RESPONSE_BATCH]

    explanations = tlm.get_explanation(
        prompt=test_prompt_batch,
        tlm_result=tlm_results,
    )

    assert isinstance(explanations, list)
    assert len(explanations) == len(test_prompt_batch)
    assert all(isinstance(exp, str) for exp in explanations)
    assert all(len(exp) > 0 for exp in explanations)

    for explanation, result in zip(explanations, tlm_results):
        assert "log" in result
        assert "explanation" in result["log"]
        assert result["log"]["explanation"] == explanation


def test_get_explanation_batch_tlm_score(tlm: TLM) -> None:
    """Test getting explanations for a batch of TLMScore objects."""

    tlm_results = [TLMScore(trustworthiness_score=0.05) for _ in range(len(test_prompt_batch))]

    explanations = tlm.get_explanation(
        prompt=test_prompt_batch,
        response=TEST_BAD_RESPONSE_BATCH,
        tlm_result=tlm_results,
    )

    assert isinstance(explanations, list)
    assert len(explanations) == len(test_prompt_batch)
    assert all(isinstance(exp, str) for exp in explanations)
    assert all(len(exp) > 0 for exp in explanations)

    for explanation, result in zip(explanations, tlm_results):
        assert "log" in result
        assert "explanation" in result["log"]
        assert result["log"]["explanation"] == explanation


def test_get_explanation_existing_explanation(tlm: TLM) -> None:
    """Test that cached explanations are returned without making API calls."""

    existing_explanation = "This explanation should be returned without making API calls."
    tlm_result = TLMResponse(
        response=TEST_BAD_RESPONSE,
        trustworthiness_score=0.1,
        log={"explanation": existing_explanation},
    )

    explanation = tlm.get_explanation(
        prompt=test_prompt,
        tlm_result=tlm_result,
    )

    assert explanation == existing_explanation
    assert tlm_result["log"]["explanation"] == existing_explanation


def test_get_explanation_missing_response_for_tlm_score(tlm: TLM) -> None:
    """Test that ValidationError is raised when response is missing for TLMScore."""

    tlm_result = TLMScore(trustworthiness_score=0.05)

    with pytest.raises(
        ValidationError,
        match="'response' is required if not provided in tlm_result",
    ):
        tlm.get_explanation(
            prompt=test_prompt,
            tlm_result=tlm_result,
        )


def test_get_explanation_extra_response_for_tlm_response(tlm: TLM) -> None:
    """Test that ValidationError is raised when response is provided for TLMResponse."""

    tlm_result = TLMResponse(
        response=TEST_BAD_RESPONSE,
        trustworthiness_score=0.1,
    )

    with pytest.raises(ValidationError, match="response should only be provided once"):
        tlm.get_explanation(
            prompt=test_prompt,
            response="Extra response",
            tlm_result=tlm_result,
        )


def test_rag_get_explanation_single_prompt(trustworthy_rag: TrustworthyRAG) -> None:
    """Test getting explanation for a single TrustworthyRAGResponse."""

    tlm_result = TrustworthyRAGResponse(
        response=TEST_BAD_RESPONSE,
        trustworthiness={"score": 0.1},
        context_sufficiency={"score": 0.1},
    )

    explanation = trustworthy_rag.get_explanation(
        query=test_query,
        context=test_context,
        tlm_result=tlm_result,
    )

    assert isinstance(explanation, str)
    assert len(explanation) > 0
    assert "trustworthiness" in tlm_result

    trustworthiness_result = cast(EvalMetric, tlm_result["trustworthiness"])
    assert "log" in trustworthiness_result
    assert "explanation" in trustworthiness_result["log"]
    assert trustworthiness_result["log"]["explanation"] == explanation


def test_rag_get_explanation_single_score(trustworthy_rag: TrustworthyRAG) -> None:
    """Test getting explanation for a single TrustworthyRAGScore."""

    tlm_result = TrustworthyRAGScore(
        trustworthiness={"score": 0.05},
        context_sufficiency={"score": 0.05},
    )

    explanation = trustworthy_rag.get_explanation(
        query=test_query,
        context=test_context,
        response=TEST_BAD_RESPONSE,
        tlm_result=tlm_result,
    )

    assert isinstance(explanation, str)
    assert len(explanation) > 0
    assert "trustworthiness" in tlm_result
    assert "log" in tlm_result["trustworthiness"]
    assert "explanation" in tlm_result["trustworthiness"]["log"]
    assert tlm_result["trustworthiness"]["log"]["explanation"] == explanation


def test_rag_get_explanation_batch_prompt(trustworthy_rag: TrustworthyRAG) -> None:
    """Test getting explanations for a batch of TrustworthyRAGResponse objects."""

    tlm_results = [
        TrustworthyRAGResponse(
            response=response,
            trustworthiness={"score": 0.1},
            context_sufficiency={"score": 0.1},
        )
        for response in TEST_BAD_RESPONSE_BATCH
    ]

    explanations = trustworthy_rag.get_explanation(
        query=test_query_batch,
        context=test_context_batch,
        tlm_result=tlm_results,
    )

    assert isinstance(explanations, list)
    assert len(explanations) == len(test_prompt_batch)
    assert all(isinstance(exp, str) for exp in explanations)
    assert all(len(exp) > 0 for exp in explanations)

    for result, explanation in zip(tlm_results, explanations):
        assert "trustworthiness" in result

        trustworthiness_result = cast(EvalMetric, result["trustworthiness"])
        assert "log" in trustworthiness_result
        assert "explanation" in trustworthiness_result["log"]
        assert trustworthiness_result["log"]["explanation"] == explanation


def test_rag_get_explanation_batch_score(trustworthy_rag: TrustworthyRAG) -> None:
    """Test getting explanations for a batch of TrustworthyRAGScore objects."""

    tlm_results = [
        TrustworthyRAGScore(
            trustworthiness={"score": 0.05},
            context_sufficiency={"score": 0.05},
        )
        for _ in range(len(test_prompt_batch))
    ]

    explanations = trustworthy_rag.get_explanation(
        query=test_query_batch,
        context=test_context_batch,
        response=TEST_BAD_RESPONSE_BATCH,
        tlm_result=tlm_results,
    )

    assert isinstance(explanations, list)
    assert len(explanations) == len(test_prompt_batch)
    assert all(isinstance(exp, str) for exp in explanations)
    assert all(len(exp) > 0 for exp in explanations)

    for result, explanation in zip(tlm_results, explanations):
        assert "trustworthiness" in result

        trustworthiness_result = result["trustworthiness"]
        assert "log" in trustworthiness_result
        assert "explanation" in trustworthiness_result["log"]
        assert trustworthiness_result["log"]["explanation"] == explanation


def test_rag_get_explanation_existing_explanation(
    trustworthy_rag: TrustworthyRAG,
) -> None:
    """Test that cached explanations are returned without making API calls."""

    existing_explanation = "This explanation should be returned without making API calls."
    tlm_result = TrustworthyRAGResponse(
        response=TEST_BAD_RESPONSE,
        trustworthiness={"score": 0.1, "log": {"explanation": existing_explanation}},
        context_sufficiency={"score": 0.1},
    )

    explanation = trustworthy_rag.get_explanation(
        query=test_query,
        context=test_context,
        tlm_result=tlm_result,
    )

    assert explanation == existing_explanation
    trustworthiness_result = cast(EvalMetric, tlm_result["trustworthiness"])
    assert trustworthiness_result["log"]["explanation"] == existing_explanation


def test_chat_completion_get_explanation_prompt(
    tlm_chat_completion: TLMChatCompletion,
) -> None:
    """Test getting explanation for a ChatCompletion response."""
    try:
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice
    except ImportError:
        pytest.skip("OpenAI not available")

    openai_kwargs = {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": test_prompt}],
    }
    tlm_result = ChatCompletion(
        id="test-id",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content=TEST_BAD_RESPONSE,
                    role="assistant",
                ),
            )
        ],
        created=1234567890,
        model="gpt-4.1-mini",
        object="chat.completion",
        tlm_metadata={"trustworthiness_score": 0.1},
    )  # type: ignore

    explanation = tlm_chat_completion.get_explanation(
        tlm_result=tlm_result,
        **openai_kwargs,  # type: ignore
    )

    assert isinstance(explanation, str)
    assert len(explanation) > 0
    tlm_metadata = tlm_result.tlm_metadata  # type: ignore
    assert "log" in tlm_metadata
    assert "explanation" in tlm_metadata["log"]
    assert tlm_metadata["log"]["explanation"] == explanation


def test_chat_completion_get_explanation_score(
    tlm_chat_completion: TLMChatCompletion,
) -> None:
    """Test getting explanation for a ChatCompletion response."""
    try:
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice
    except ImportError:
        pytest.skip("OpenAI not available")

    openai_kwargs = {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": test_prompt}],
    }
    response = ChatCompletion(
        id="test-id",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content=TEST_BAD_RESPONSE,
                    role="assistant",
                ),
            )
        ],
        created=1234567890,
        model="gpt-4.1-mini",
        object="chat.completion",
    )
    tlm_result = TLMScore(trustworthiness_score=0.05)

    explanation = tlm_chat_completion.get_explanation(
        response=response,
        tlm_result=tlm_result,
        **openai_kwargs,
    )

    assert isinstance(explanation, str)
    assert "log" in tlm_result
    assert "explanation" in tlm_result["log"]
    assert tlm_result["log"]["explanation"] == explanation


def test_chat_completion_get_explanation_prompt_existing_explanation(
    tlm_chat_completion: TLMChatCompletion,
) -> None:
    """Test getting explanation for a ChatCompletion response."""
    try:
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice
    except ImportError:
        pytest.skip("OpenAI not available")

    existing_explanation = "This explanation should be returned without making API calls."

    openai_kwargs = {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": test_prompt}],
    }
    tlm_result = ChatCompletion(
        id="test-id",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content=TEST_BAD_RESPONSE,
                    role="assistant",
                ),
            )
        ],
        created=1234567890,
        model="gpt-4.1-mini",
        object="chat.completion",
        tlm_metadata={
            "trustworthiness_score": 0.1,
            "log": {"explanation": existing_explanation},
        },
    )  # type: ignore

    explanation = tlm_chat_completion.get_explanation(
        tlm_result=tlm_result,
        **openai_kwargs,  # type: ignore
    )

    assert explanation == existing_explanation
    assert tlm_result.tlm_metadata["log"]["explanation"] == existing_explanation  # type: ignore


def test_chat_completion_get_explanation_score_existing_explanation(
    tlm_chat_completion: TLMChatCompletion,
) -> None:
    """Test getting explanation for a ChatCompletion response."""
    try:
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice
    except ImportError:
        pytest.skip("OpenAI not available")

    existing_explanation = "This explanation should be returned without making API calls."

    openai_kwargs = {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": test_prompt}],
    }
    response = ChatCompletion(
        id="test-id",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content=TEST_BAD_RESPONSE,
                    role="assistant",
                ),
            )
        ],
        created=1234567890,
        model="gpt-4.1-mini",
        object="chat.completion",
    )
    tlm_result = TLMScore(trustworthiness_score=0.05, log={"explanation": existing_explanation})

    explanation = tlm_chat_completion.get_explanation(
        response=response,
        tlm_result=tlm_result,
        **openai_kwargs,
    )

    assert explanation == existing_explanation
    assert tlm_result["log"]["explanation"] == existing_explanation

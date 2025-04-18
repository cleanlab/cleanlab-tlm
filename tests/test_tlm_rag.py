import os
from collections.abc import Generator
from typing import Any, cast
from unittest import mock

import pytest

from cleanlab_tlm.errors import APITimeoutError, MissingApiKeyError, ValidationError
from cleanlab_tlm.internal.api import api
from cleanlab_tlm.internal.constants import _TLM_DEFAULT_MODEL
from cleanlab_tlm.tlm import TLMOptions
from cleanlab_tlm.utils.rag import (
    _DEFAULT_EVALS,
    Eval,
    EvalMetric,
    TrustworthyRAG,
    TrustworthyRAGResponse,
    TrustworthyRAGScore,
    get_default_evals,
)
from tests.conftest import make_text_unique
from tests.constants import TEST_PROMPT

# Test constants
TEST_QUERY = "What is the capital of France?"
TEST_CONTEXT = "France is a country in Western Europe. Its capital is Paris, which is known for the Eiffel Tower."
TEST_RESPONSE = "The capital of France is Paris."

# Make unique test data to avoid caching issues
test_query = make_text_unique(TEST_QUERY)
test_context = make_text_unique(TEST_CONTEXT)
test_response = make_text_unique(TEST_RESPONSE)
test_prompt = make_text_unique(TEST_PROMPT)

# Create test batches
test_query_batch = [
    make_text_unique("What is the capital of France?"),
    make_text_unique("What is the capital of Germany?"),
]
test_context_batch = [
    make_text_unique(
        "France is a country in Western Europe. Its capital is Paris, which is known for the Eiffel Tower."
    ),
    make_text_unique("Germany is a country in Central Europe. Its capital is Berlin, known for the Brandenburg Gate."),
]
test_response_batch = [
    make_text_unique("The capital of France is Paris."),
    make_text_unique("The capital of Germany is Berlin."),
]
test_prompt_batch = [
    make_text_unique("Using the context information, answer the following question: What is the capital of France?"),
    make_text_unique("Using the context information, answer the following question: What is the capital of Germany?"),
]


@pytest.fixture(scope="module")
def trustworthy_rag_api_key() -> str:
    api_key = os.environ.get("CLEANLAB_TLM_API_KEY")
    if api_key is None:
        raise MissingApiKeyError
    return api_key


@pytest.fixture(scope="module")
def trustworthy_rag(trustworthy_rag_api_key: str) -> TrustworthyRAG:
    try:
        return TrustworthyRAG(api_key=trustworthy_rag_api_key)
    except Exception as e:
        environment = os.environ.get("CLEANLAB_API_BASE_URL")
        pytest.skip(f"Failed to create TrustworthyRAG: {e}. Check your API key and environment: ({environment}).")


def is_trustworthy_rag_response(response: Any) -> bool:
    """Check if an object is a valid TrustworthyRAGResponse."""
    if response is None:
        return False

    if isinstance(response, dict) and "response" in response:
        # Check for the presence of at least one evaluation metric
        has_evaluation_metrics = False
        for key, value in response.items():
            if key != "response":
                if not isinstance(value, dict) or "score" not in value:
                    return False

                score = value["score"]
                if score is not None and not (isinstance(score, float) and 0.0 <= score <= 1.0):
                    return False

                has_evaluation_metrics = True

        return has_evaluation_metrics

    return False


def is_trustworthy_rag_score(score: Any) -> bool:
    """Check if an object is a valid TrustworthyRAGScore."""
    if score is None:
        return False

    if isinstance(score, dict):
        # Should have at least one evaluation metric
        if len(score) == 0:
            return False

        for metric_data in score.values():
            if not isinstance(metric_data, dict) or "score" not in metric_data:
                return False

            score_value = metric_data["score"]
            if score_value is not None and not (isinstance(score_value, float) and 0.0 <= score_value <= 1.0):
                return False

        return True

    return False


def test_init_with_api_key(trustworthy_rag_api_key: str) -> None:
    rag = TrustworthyRAG(api_key=trustworthy_rag_api_key)

    assert rag is not None
    assert rag._api_key == trustworthy_rag_api_key
    assert rag._quality_preset == "medium"
    assert rag._options == {"model": _TLM_DEFAULT_MODEL}
    assert len(rag._evals) > 0


def test_init_with_missing_api_key() -> None:
    original_api_key = os.environ.get("CLEANLAB_TLM_API_KEY")
    if original_api_key:
        os.environ.pop("CLEANLAB_TLM_API_KEY")

    try:
        with pytest.raises(MissingApiKeyError):
            TrustworthyRAG()
    finally:
        if original_api_key:
            os.environ["CLEANLAB_TLM_API_KEY"] = original_api_key


def test_init_with_custom_evals(trustworthy_rag_api_key: str) -> None:
    custom_evals = [
        Eval(
            name="test_evaluation",
            criteria="Evaluate the response based on X",
            query_identifier="Question",
            context_identifier="Context",
            response_identifier="Answer",
        )
    ]

    rag = TrustworthyRAG(api_key=trustworthy_rag_api_key, evals=custom_evals)

    assert rag is not None
    assert len(rag._evals) == 1
    assert rag._evals[0].name == "test_evaluation"
    assert rag._evals[0].criteria == "Evaluate the response based on X"
    assert rag._evals[0].query_identifier == "Question"
    assert rag._evals[0].context_identifier == "Context"
    assert rag._evals[0].response_identifier == "Answer"


def test_init_with_empty_evals(trustworthy_rag_api_key: str) -> None:
    """Tests TrustworthyRAG initialization with an empty list of evaluations."""
    empty_evals: list[Eval] = []

    rag = TrustworthyRAG(api_key=trustworthy_rag_api_key, evals=empty_evals)

    assert rag is not None
    assert rag._evals == []
    assert len(rag._evals) == 0


def test_init_with_options(trustworthy_rag_api_key: str) -> None:
    max_tokens = 500

    options: TLMOptions = {
        "model": "gpt-4",
        "max_tokens": max_tokens,
        "use_self_reflection": True,
    }

    rag = TrustworthyRAG(api_key=trustworthy_rag_api_key, options=options)

    assert rag is not None
    assert rag._options is not None
    assert rag._options["model"] == "gpt-4"
    assert rag._options["max_tokens"] == max_tokens
    assert rag._options["use_self_reflection"] is True


def test_init_with_quality_preset(trustworthy_rag_api_key: str) -> None:
    rag_medium = TrustworthyRAG(quality_preset="medium", api_key=trustworthy_rag_api_key)
    assert rag_medium is not None
    assert rag_medium._quality_preset == "medium"

    rag_low = TrustworthyRAG(quality_preset="low", api_key=trustworthy_rag_api_key)
    assert rag_low is not None
    assert rag_low._quality_preset == "low"

    with pytest.raises(ValidationError):
        TrustworthyRAG(quality_preset="high", api_key=trustworthy_rag_api_key)


def test_get_evals(trustworthy_rag: TrustworthyRAG) -> None:
    evals = trustworthy_rag.get_evals()

    assert evals is not None
    assert len(evals) > 0
    assert all(isinstance(eval_obj, Eval) for eval_obj in evals)

    # Verify that the returned list is a copy
    original_evals = trustworthy_rag._evals
    evals.append(Eval(name="new_eval", criteria="New criteria", query_identifier="query"))
    assert len(evals) == len(original_evals) + 1
    assert len(trustworthy_rag._evals) == len(original_evals)


def test_get_default_evals() -> None:
    evals = get_default_evals()

    assert evals is not None
    assert len(evals) > 0
    assert all(isinstance(eval_obj, Eval) for eval_obj in evals)

    # Verify that default evals include specific expected evaluations
    eval_names = [eval_obj.name for eval_obj in evals]
    expected_evals = [eval_config["name"] for eval_config in _DEFAULT_EVALS]
    for expected_eval in expected_evals:
        assert expected_eval in eval_names


def test_eval_class_initialization() -> None:
    eval_obj = Eval(
        name="test_eval",
        criteria="Test evaluation criteria",
        query_identifier="Query",
        context_identifier="Context",
        response_identifier="Response",
    )

    assert eval_obj is not None
    assert eval_obj.name == "test_eval"
    assert eval_obj.criteria == "Test evaluation criteria"
    assert eval_obj.query_identifier == "Query"
    assert eval_obj.context_identifier == "Context"
    assert eval_obj.response_identifier == "Response"


def test_eval_class_with_defaults() -> None:
    eval_obj = Eval(
        name="test_eval",
        criteria="Test evaluation criteria",
        query_identifier="Query",  # Adding at least one identifier to pass validation
    )

    assert eval_obj is not None
    assert eval_obj.name == "test_eval"
    assert eval_obj.criteria == "Test evaluation criteria"
    assert eval_obj.query_identifier == "Query"
    assert eval_obj.context_identifier is None
    assert eval_obj.response_identifier is None


def test_eval_class_requires_at_least_one_identifier() -> None:
    """Test that creating an Eval without any identifiers raises a ValueError."""
    with pytest.raises(
        ValueError,
        match="At least one of query_identifier, context_identifier, or response_identifier must be specified.",
    ):
        Eval(
            name="test_eval",
            criteria="Test evaluation criteria",
            query_identifier=None,
            context_identifier=None,
            response_identifier=None,
        )

    # Also test the default case where all identifiers default to None
    with pytest.raises(
        ValueError,
        match="At least one of query_identifier, context_identifier, or response_identifier must be specified.",
    ):
        Eval(
            name="test_eval",
            criteria="Test evaluation criteria",
        )


def test_custom_eval_criteria_not_supported(trustworthy_rag_api_key: str) -> None:
    """Tests that custom_eval_criteria is not supported in TrustworthyRAG."""
    custom_eval_criteria = [{"name": "test_criteria", "criteria": "This is a test criteria"}]

    with pytest.raises(
        ValidationError,
        match="^custom_eval_criteria is not supported for this class$",
    ):
        TrustworthyRAG(api_key=trustworthy_rag_api_key, options=TLMOptions(custom_eval_criteria=custom_eval_criteria))


def test_evaluation_metric_type() -> None:
    score_high = 0.85
    score_medium = 0.75

    metric: EvalMetric = {
        "score": score_high,
    }

    assert metric is not None
    assert "score" in metric
    assert metric["score"] == score_high

    metric_with_log: EvalMetric = {
        "score": score_medium,
        "log": {"explanation": "This is a test explanation"},
    }

    assert metric_with_log is not None
    assert "score" in metric_with_log
    assert metric_with_log["score"] == score_medium
    assert "log" in metric_with_log
    assert metric_with_log["log"] == {"explanation": "This is a test explanation"}

    metric_with_none_score: EvalMetric = {
        "score": None,
    }

    assert metric_with_none_score is not None
    assert "score" in metric_with_none_score
    assert metric_with_none_score["score"] is None


def test_trustworthy_rag_response_type() -> None:
    trustworthiness_score = 0.92
    context_score = 0.9
    response_score = 0.85

    response = TrustworthyRAGResponse(
        {
            "response": "This is a test response",
        }
    )

    assert response is not None
    assert "response" in response
    assert response["response"] == "This is a test response"

    response_with_metrics = TrustworthyRAGResponse(
        {
            "response": "This is a test response",
            "trustworthiness": {
                "score": trustworthiness_score,
                "log": {"explanation": "Did not find a reason to doubt trustworthiness."},
            },
            "context_informativeness": {"score": context_score},
            "response_helpfulness": {"score": response_score},
        }
    )

    assert response_with_metrics is not None
    assert "response" in response_with_metrics
    assert response_with_metrics["response"] == "This is a test response"
    assert "trustworthiness" in response_with_metrics
    assert "context_informativeness" in response_with_metrics
    assert "response_helpfulness" in response_with_metrics
    assert cast(dict[str, Any], response_with_metrics["trustworthiness"])["score"] == trustworthiness_score
    assert cast(dict[str, Any], response_with_metrics["context_informativeness"])["score"] == context_score
    assert cast(dict[str, Any], response_with_metrics["response_helpfulness"])["score"] == response_score

    # Test that the response is properly detected by our validation function
    assert is_trustworthy_rag_response(response_with_metrics)

    response_with_none = TrustworthyRAGResponse(
        {
            "response": None,
        }
    )

    assert response_with_none is not None
    assert "response" in response_with_none
    assert response_with_none["response"] is None


def test_trustworthy_rag_score_type() -> None:
    context_score = 0.9
    response_score = 0.85

    score = TrustworthyRAGScore(
        {
            "context_informativeness": {"score": context_score},
            "response_helpfulness": {"score": response_score},
        }
    )

    assert score is not None
    assert "context_informativeness" in score
    assert "response_helpfulness" in score
    assert cast(dict[str, Any], score["context_informativeness"])["score"] == context_score
    assert cast(dict[str, Any], score["response_helpfulness"])["score"] == response_score

    # Test that the score is properly detected by our validation function
    assert is_trustworthy_rag_score(score)

    score_with_logs = TrustworthyRAGScore(
        {
            "context_informativeness": {
                "score": context_score,
                "log": {"explanation": "The context is very informative"},
            },
            "response_helpfulness": {
                "score": response_score,
                "log": {"explanation": "The response is helpful"},
            },
        }
    )

    assert score_with_logs is not None
    assert "context_informativeness" in score_with_logs
    assert "response_helpfulness" in score_with_logs
    assert cast(dict[str, Any], score_with_logs["context_informativeness"])["score"] == context_score
    assert (
        cast(dict[str, Any], cast(dict[str, Any], score_with_logs["context_informativeness"])["log"])["explanation"]
        == "The context is very informative"
    )
    assert cast(dict[str, Any], score_with_logs["response_helpfulness"])["score"] == response_score
    assert (
        cast(dict[str, Any], cast(dict[str, Any], score_with_logs["response_helpfulness"])["log"])["explanation"]
        == "The response is helpful"
    )

    score_with_empty_scores = TrustworthyRAGScore({})

    assert score_with_empty_scores is not None
    assert len(score_with_empty_scores) == 0

    # Test that empty score dict is not a valid score
    assert not is_trustworthy_rag_score(score_with_empty_scores)


def test_generate_with_query_and_context(trustworthy_rag: TrustworthyRAG) -> None:
    response = trustworthy_rag.generate(
        query=test_query,
        context=test_context,
    )

    assert response is not None
    assert is_trustworthy_rag_response(response)
    assert "response" in response
    assert cast(dict[str, Any], response)["response"] is not None

    # Check for required evaluation metrics in the response
    assert any(k for k in response if k != "response")


def test_generate_with_prompt(trustworthy_rag: TrustworthyRAG) -> None:
    response = trustworthy_rag.generate(
        query=test_query,
        context=test_context,
        prompt=test_prompt,
    )

    assert response is not None
    assert is_trustworthy_rag_response(response)
    assert "response" in response
    assert cast(dict[str, Any], response)["response"] is not None


def test_generate_with_custom_form_prompt(trustworthy_rag: TrustworthyRAG) -> None:
    def custom_form_prompt(query: str, context: str) -> str:
        system_prompt = "You are a helpful assistant that provides accurate information based on the context."
        prompt = f"{system_prompt}\n\n"
        prompt += f"CUSTOM PROMPT FORMAT\n\nQUESTION: {query}\n\nINFORMATION: {context}\n\n"
        prompt += "ANSWER:"
        return prompt

    response = trustworthy_rag.generate(
        query=test_query,
        context=test_context,
        form_prompt=custom_form_prompt,
    )

    assert response is not None
    assert is_trustworthy_rag_response(response)
    assert "response" in response
    assert cast(dict[str, Any], response)["response"] is not None


def test_generate_with_empty_evals(trustworthy_rag_api_key: str) -> None:
    """Tests RAG generate with empty evaluations list."""
    # Create a TrustworthyRAG instance with empty evals
    rag = TrustworthyRAG(api_key=trustworthy_rag_api_key, evals=[])

    # Generate response with empty evals
    response = rag.generate(
        query=test_query,
        context=test_context,
    )

    # Check for response with minimal evaluation metrics
    assert response is not None
    assert is_trustworthy_rag_response(response)
    assert "response" in response
    assert cast(dict[str, Any], response)["response"] is not None

    # Since we have empty evals, there should be minimal or no evaluation metrics
    # other than potentially a trustworthiness score which might be included by default
    available_keys = set(cast(dict[str, Any], response).keys())
    assert available_keys.issubset({"response", "trustworthiness"})


def test_default_prompt_formatter() -> None:
    """Test that the default prompt formatter works as expected"""
    formatted_prompt = TrustworthyRAG._default_prompt_formatter(
        query=test_query,
        context=test_context,
    )

    assert formatted_prompt is not None
    assert isinstance(formatted_prompt, str)
    assert test_query in formatted_prompt
    assert test_context in formatted_prompt
    assert "Context information is below." in formatted_prompt
    assert "User: " in formatted_prompt
    assert "Assistant: " in formatted_prompt


def test_generate_missing_required_params(trustworthy_rag: TrustworthyRAG) -> None:
    with pytest.raises(ValidationError):
        trustworthy_rag.generate(query=test_query, context=None)  # type: ignore

    with pytest.raises(ValidationError):
        trustworthy_rag.generate(query=None, context=test_context)  # type: ignore

    with pytest.raises(ValidationError):
        trustworthy_rag.generate(query=None, context=None)  # type: ignore


def test_score_with_query_context_response(trustworthy_rag: TrustworthyRAG) -> None:
    score = trustworthy_rag.score(
        query=test_query,
        context=test_context,
        response=test_response,
    )

    assert score is not None
    assert is_trustworthy_rag_score(score)
    assert len(score) > 0
    for metric_name, metric_data in cast(dict[str, Any], score).items():
        assert "score" in metric_data
        assert metric_name  # Ensure non-empty metric name


def test_score_with_prompt_and_response(trustworthy_rag: TrustworthyRAG) -> None:
    score = trustworthy_rag.score(
        prompt=test_prompt,
        response=test_response,
        query=test_query,
        context=test_context,
    )

    assert score is not None
    assert is_trustworthy_rag_score(score)
    assert len(score) > 0
    for metric_data in cast(dict[str, Any], score).values():
        assert "score" in metric_data


def test_score_with_custom_form_prompt(trustworthy_rag: TrustworthyRAG) -> None:
    def custom_form_prompt(query: str, context: str) -> str:
        system_prompt = "You are a helpful assistant that provides accurate information based on the context."
        prompt = f"{system_prompt}\n\n"
        prompt += f"CUSTOM FORMAT\n\nQUESTION: {query}\n\nCONTEXT: {context}\n\n"
        prompt += "ANSWER:"
        return prompt

    score = trustworthy_rag.score(
        response=test_response,
        query=test_query,
        context=test_context,
        form_prompt=custom_form_prompt,
    )

    assert score is not None
    assert is_trustworthy_rag_score(score)


def test_score_with_empty_evals(trustworthy_rag_api_key: str) -> None:
    """Tests RAG score with empty evaluations list."""
    # Create a TrustworthyRAG instance with empty evals
    rag = TrustworthyRAG(api_key=trustworthy_rag_api_key, evals=[])

    # Score response with empty evals
    score = rag.score(
        response=test_response,
        query=test_query,
        context=test_context,
    )

    # Check the score
    assert score is not None
    assert is_trustworthy_rag_score(score)

    # Since we have empty evals, there should be minimal or no evaluation metrics
    # other than potentially a trustworthiness score which might be included by default
    available_keys = set(cast(dict[str, Any], score).keys())
    assert available_keys.issubset({"trustworthiness"})


@pytest.mark.asyncio
async def test_api_tlm_rag_generate_empty_evals() -> None:
    """Test the API level handling of empty evals for tlm_rag_generate."""
    # Sample test data
    test_api_key = "test_api_key"
    test_prompt = "What is the capital of France?"
    test_query = "Capital of France?"
    test_context = "France is in Europe and its capital is Paris."
    test_quality_preset = "medium"
    mock_trustworthiness_score = 0.95

    # Mock response data
    mock_response_data = {
        "response": "The capital of France is Paris.",
        "trustworthiness": {"score": mock_trustworthiness_score},
    }

    # Create mock session and response
    mock_response = mock.MagicMock()
    mock_response.status = 200
    mock_response.json = mock.AsyncMock(return_value=mock_response_data)

    mock_session = mock.MagicMock()
    mock_session.post = mock.AsyncMock(return_value=mock_response)
    mock_session.close = mock.AsyncMock()

    # Create mock rate handler
    mock_rate_handler = mock.MagicMock()
    mock_rate_handler.__aenter__ = mock.AsyncMock()
    mock_rate_handler.__aexit__ = mock.AsyncMock()

    # Patch client session and test API call
    with mock.patch("aiohttp.ClientSession", return_value=mock_session):
        result = await api.tlm_rag_generate(
            api_key=test_api_key,
            prompt=test_prompt,
            query=test_query,
            context=test_context,
            evals=[],  # Empty evals list
            quality_preset=test_quality_preset,
            options=None,
            rate_handler=mock_rate_handler,
        )

    # Verify the result
    assert result is not None
    assert "response" in result
    assert result["response"] == "The capital of France is Paris."
    assert "trustworthiness" in result
    assert result["trustworthiness"]["score"] == mock_trustworthiness_score

    # Verify the API call
    mock_session.post.assert_called_once()
    call_args = mock_session.post.call_args

    # Extract the JSON payload
    json_payload = call_args[1]["json"]

    # Verify the evals parameter was passed correctly as an empty list
    assert "evals" in json_payload
    assert json_payload["evals"] == []


@pytest.mark.asyncio
async def test_api_tlm_rag_score_empty_evals() -> None:
    """Test the API level handling of empty evals for tlm_rag_score."""
    # Sample test data
    test_api_key = "test_api_key"
    test_prompt = "What is the capital of France?"
    test_query = "Capital of France?"
    test_context = "France is in Europe and its capital is Paris."
    test_response = "The capital of France is Paris."
    test_quality_preset = "medium"
    mock_trustworthiness_score = 0.95

    # Mock response data
    mock_response_data = {"trustworthiness": {"score": mock_trustworthiness_score}}

    # Create mock session and response
    mock_response = mock.MagicMock()
    mock_response.status = 200
    mock_response.json = mock.AsyncMock(return_value=mock_response_data)

    mock_session = mock.MagicMock()
    mock_session.post = mock.AsyncMock(return_value=mock_response)
    mock_session.close = mock.AsyncMock()

    # Create mock rate handler
    mock_rate_handler = mock.MagicMock()
    mock_rate_handler.__aenter__ = mock.AsyncMock()
    mock_rate_handler.__aexit__ = mock.AsyncMock()

    # Patch client session and test API call
    with mock.patch("aiohttp.ClientSession", return_value=mock_session):
        result = await api.tlm_rag_score(
            api_key=test_api_key,
            response={"response": test_response},
            prompt=test_prompt,
            query=test_query,
            context=test_context,
            evals=[],  # Empty evals list
            quality_preset=test_quality_preset,
            options=None,
            rate_handler=mock_rate_handler,
        )

    # Verify the result
    assert result is not None
    assert "trustworthiness" in result
    assert result["trustworthiness"]["score"] == mock_trustworthiness_score

    # Verify the API call
    mock_session.post.assert_called_once()
    call_args = mock_session.post.call_args

    # Extract the JSON payload
    json_payload = call_args[1]["json"]

    # Verify the evals parameter was passed correctly as an empty list
    assert "evals" in json_payload
    assert json_payload["evals"] == []


def test_score_missing_required_params(trustworthy_rag: TrustworthyRAG) -> None:
    with pytest.raises(ValidationError):
        trustworthy_rag.score(
            response=None,  # type: ignore
            query=test_query,
            context=test_context,
        )

    with pytest.raises(ValidationError):
        trustworthy_rag.score(
            response=test_response,
            query=None,  # type: ignore
            context=None,  # type: ignore
        )

    with pytest.raises(ValidationError):
        trustworthy_rag.score(response=test_response, query=test_query, context=None)  # type: ignore

    with pytest.raises(ValidationError):
        trustworthy_rag.score(response=test_response, query=None, context=test_context)  # type: ignore


def test_generate_with_prompt_and_form_prompt(trustworthy_rag: TrustworthyRAG) -> None:
    def custom_form_prompt(query: str, context: str) -> str:
        return f"Custom prompt with query: {query} and context: {context}"

    with pytest.raises(ValidationError):
        trustworthy_rag.generate(
            query=test_query,
            context=test_context,
            prompt=test_prompt,
            form_prompt=custom_form_prompt,
        )


def test_score_with_prompt_and_form_prompt(trustworthy_rag: TrustworthyRAG) -> None:
    def custom_form_prompt(query: str, context: str) -> str:
        return f"Question: {query}\nContext: {context}"

    with pytest.raises(ValidationError):
        trustworthy_rag.score(
            response=test_response,
            query=test_query,
            context=test_context,
            prompt=test_prompt,
            form_prompt=custom_form_prompt,
        )


def test_init_with_unsupported_quality_preset(trustworthy_rag_api_key: str) -> None:
    with pytest.raises(ValidationError):
        TrustworthyRAG(
            quality_preset="unsupported_preset",  # type: ignore
            api_key=trustworthy_rag_api_key,
        )


def test_batch_generate(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests batch generate functionality of TrustworthyRAG.

    Expected:
    - TrustworthyRAG should return a list of TrustworthyRAGResponse objects
    - Each response should be a valid TrustworthyRAGResponse object
    """
    # act
    responses = trustworthy_rag.generate(
        query=test_query_batch,
        context=test_context_batch,
    )

    # assert
    assert isinstance(responses, list)
    assert all(is_trustworthy_rag_response(r) for r in responses)


def test_batch_score(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests batch score functionality of TrustworthyRAG.

    Expected:
    - TrustworthyRAG should return a list of TrustworthyRAGScore objects
    - Each score should be a valid TrustworthyRAGScore object
    """
    # act
    scores = trustworthy_rag.score(
        query=test_query_batch,
        context=test_context_batch,
        response=test_response_batch,
    )

    # assert
    assert isinstance(scores, list)
    assert all(is_trustworthy_rag_score(s) for s in scores)


def test_batch_generate_with_prompt(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests batch generate functionality of TrustworthyRAG with prompt.

    Expected:
    - TrustworthyRAG should return a list of TrustworthyRAGResponse objects
    - Each response should be a valid TrustworthyRAGResponse object
    """
    # act
    responses = trustworthy_rag.generate(
        query=test_query_batch,
        context=test_context_batch,
        prompt=test_prompt_batch,
    )

    # assert
    assert isinstance(responses, list)
    assert all(is_trustworthy_rag_response(r) for r in responses)


def test_batch_score_with_prompt(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests batch score functionality of TrustworthyRAG with prompt.

    Expected:
    - TrustworthyRAG should return a list of TrustworthyRAGScore objects
    - Each score should be a valid TrustworthyRAGScore object
    """
    # act
    scores = trustworthy_rag.score(
        query=test_query_batch,
        context=test_context_batch,
        response=test_response_batch,
        prompt=test_prompt_batch,
    )

    # assert
    assert isinstance(scores, list)
    assert all(is_trustworthy_rag_score(s) for s in scores)


def test_batch_generate_with_custom_form_prompt(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests batch generate functionality of TrustworthyRAG with custom form_prompt.

    Expected:
    - TrustworthyRAG should return a list of TrustworthyRAGResponse objects
    - Each response should be a valid TrustworthyRAGResponse object
    """

    def custom_form_prompt(query: str, context: str) -> str:
        return f"System: Always be helpful\nContext: {context}\nUser: {query}"

    # act
    responses = trustworthy_rag.generate(
        query=test_query_batch,
        context=test_context_batch,
        form_prompt=custom_form_prompt,
    )

    # assert
    assert isinstance(responses, list)
    assert all(is_trustworthy_rag_response(r) for r in responses)


def test_batch_score_with_custom_form_prompt(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests batch score functionality of TrustworthyRAG with custom form_prompt.

    Expected:
    - TrustworthyRAG should return a list of TrustworthyRAGScore objects
    - Each score should be a valid TrustworthyRAGScore object
    """

    def custom_form_prompt(query: str, context: str) -> str:
        return f"System: Always be helpful\nContext: {context}\nUser: {query}"

    # act
    scores = trustworthy_rag.score(
        query=test_query_batch,
        context=test_context_batch,
        response=test_response_batch,
        form_prompt=custom_form_prompt,
    )

    # assert
    assert isinstance(scores, list)
    assert all(is_trustworthy_rag_score(s) for s in scores)


def test_generate_force_timeouts(trustworthy_rag: TrustworthyRAG, reset_rag_timeout: None) -> None:  # noqa: ARG001
    """Tests single prompt generate with forced timeouts.

    Sets timeout to 0.0001 seconds, which should force a timeout error being thrown.

    Expected:
    - TrustworthyRAG should raise a timeout error
    """
    # arrange -- override timeout
    trustworthy_rag._timeout = 0.0001

    # assert -- timeout is thrown
    with pytest.raises(APITimeoutError):
        # act -- run a generate
        trustworthy_rag.generate(
            query=test_query,
            context=test_context,
        )


def test_score_force_timeouts(trustworthy_rag: TrustworthyRAG, reset_rag_timeout: None) -> None:  # noqa: ARG001
    """Tests single score with forced timeouts.

    Sets timeout to 0.0001 seconds, which should force a timeout error being thrown.

    Expected:
    - TrustworthyRAG should raise a timeout error
    """
    # arrange -- override timeout
    trustworthy_rag._timeout = 0.0001

    # assert -- timeout is thrown
    with pytest.raises(APITimeoutError):
        # act -- run a batch score
        trustworthy_rag.score(
            query=test_query,
            context=test_context,
            response=test_response,
        )


@pytest.fixture
def reset_rag_timeout(trustworthy_rag: TrustworthyRAG) -> Generator[None, None, None]:
    """Reset the timeout on the TrustworthyRAG fixture after tests that modify it."""
    old_timeout = trustworthy_rag._timeout
    yield
    trustworthy_rag._timeout = old_timeout

import os
from typing import Any

import pytest

from cleanlab_tlm.errors import MissingApiKeyError, ValidationError
from cleanlab_tlm.tlm import TLMOptions
from cleanlab_tlm.utils.tlm_rag import (
    Eval,
    EvaluationMetric,
    TrustworthyRAG,
    TrustworthyRAGResponse,
    TrustworthyRAGScore,
    get_default_evals,
)
from tests.conftest import make_text_unique
from tests.constants import TEST_PROMPT

# Test constants for TrustworthyRAG
TEST_QUERY = "What is the capital of France?"
TEST_CONTEXT = "France is a country in Western Europe. Its capital is Paris, which is known for the Eiffel Tower."
TEST_RESPONSE = "The capital of France is Paris."

# Make unique test data to avoid caching issues
test_query = make_text_unique(TEST_QUERY)
test_context = make_text_unique(TEST_CONTEXT)
test_response = make_text_unique(TEST_RESPONSE)
test_prompt = make_text_unique(TEST_PROMPT)


@pytest.fixture(scope="module")
def trustworthy_rag_api_key() -> str:
    api_key = os.environ.get("CLEANLAB_TLM_API_KEY")
    if api_key is None:
        raise MissingApiKeyError
    return api_key


@pytest.fixture(scope="module")
def trustworthy_rag(trustworthy_rag_api_key: str) -> TrustworthyRAG:
    """Creates a TrustworthyRAG with default settings."""
    try:
        # uses environment API key
        return TrustworthyRAG(api_key=trustworthy_rag_api_key)
    except Exception as e:
        environment = os.environ.get("CLEANLAB_API_BASE_URL")
        pytest.skip(f"Failed to create TrustworthyRAG: {e}. Check your API key and environment: ({environment}).")


def is_trustworthy_rag_response(response: Any) -> bool:
    """Returns True if the response is a TrustworthyRAGResponse."""
    if response is None:
        return False

    if isinstance(response, dict) and "response" in response:
        # Check each key that is not "response" as a potential evaluation metric
        for key, value in response.items():
            if key != "response":
                if not isinstance(value, dict) or "score" not in value:
                    return False

                score = value["score"]
                if score is not None and not (isinstance(score, float) and 0.0 <= score <= 1.0):
                    return False

        return True

    return False


def is_trustworthy_rag_score(score: Any) -> bool:
    """Returns True if the score is a TrustworthyRAGScore."""
    if score is None:
        return False

    if isinstance(score, dict):
        # TrustworthyRAGScore has evaluation metrics as top-level keys
        # Each metric should be a dict with a "score" key
        for metric_name, metric_data in score.items():
            if not isinstance(metric_data, dict) or "score" not in metric_data:
                return False

            score_value = metric_data["score"]
            if score_value is not None and not (isinstance(score_value, float) and 0.0 <= score_value <= 1.0):
                return False

        # At least one valid metric should be present
        return len(score) > 0

    return False


def test_init_with_api_key(trustworthy_rag_api_key: str) -> None:
    """Tests initializing TrustworthyRAG with an API key."""
    # Act
    rag = TrustworthyRAG(api_key=trustworthy_rag_api_key)

    # Assert
    assert rag is not None
    assert rag._api_key == trustworthy_rag_api_key
    assert rag._quality_preset == "medium"  # Default quality preset
    assert rag._options is None  # Default options
    assert len(rag._evals) > 0  # Default evals should be loaded


def test_init_with_missing_api_key() -> None:
    """Tests initializing TrustworthyRAG with a missing API key."""
    # Temporarily clear the environment variable
    original_api_key = os.environ.get("CLEANLAB_TLM_API_KEY")
    if original_api_key:
        os.environ.pop("CLEANLAB_TLM_API_KEY")

    try:
        # Act & Assert
        with pytest.raises(MissingApiKeyError):
            TrustworthyRAG()
    finally:
        # Restore the environment variable
        if original_api_key:
            os.environ["CLEANLAB_TLM_API_KEY"] = original_api_key


def test_init_with_custom_evals(trustworthy_rag_api_key: str) -> None:
    """Tests initializing TrustworthyRAG with custom evals."""
    # Arrange
    custom_evals = [
        Eval(
            name="custom_eval",
            criteria="Custom evaluation criteria",
            query_identifier="Q",
            context_identifier="C",
            response_identifier="R",
        )
    ]

    # Act
    rag = TrustworthyRAG(api_key=trustworthy_rag_api_key, evals=custom_evals)

    # Assert
    assert rag is not None
    assert len(rag._evals) == 1
    assert rag._evals[0].name == "custom_eval"
    assert rag._evals[0].criteria == "Custom evaluation criteria"
    assert rag._evals[0].query_identifier == "Q"
    assert rag._evals[0].context_identifier == "C"
    assert rag._evals[0].response_identifier == "R"


def test_init_with_options(trustworthy_rag_api_key: str) -> None:
    """Tests initializing TrustworthyRAG with custom options."""
    # Arrange
    options = TLMOptions(
        model="gpt-4",
        max_tokens=500,
        use_self_reflection=True,
    )

    # Act
    rag = TrustworthyRAG(api_key=trustworthy_rag_api_key, options=options)

    # Assert
    assert rag is not None
    assert rag._options is not None
    print(rag._options)
    assert rag._options["model"] == "gpt-4"
    assert rag._options["max_tokens"] == 500
    assert rag._options["use_self_reflection"] is True


def test_init_with_quality_preset(trustworthy_rag_api_key: str) -> None:
    """Tests initializing TrustworthyRAG with a custom quality preset."""
    # Act & Assert - Test valid presets
    rag_medium = TrustworthyRAG(quality_preset="medium", api_key=trustworthy_rag_api_key)
    assert rag_medium is not None
    assert rag_medium._quality_preset == "medium"

    rag_low = TrustworthyRAG(quality_preset="low", api_key=trustworthy_rag_api_key)
    assert rag_low is not None
    assert rag_low._quality_preset == "low"

    # Act & Assert - Test invalid preset
    with pytest.raises(ValidationError):
        TrustworthyRAG(quality_preset="high", api_key=trustworthy_rag_api_key)


def test_get_evals(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests getting evals from TrustworthyRAG."""
    # Act
    evals = trustworthy_rag.get_evals()

    # Assert
    assert evals is not None
    assert len(evals) > 0
    assert all(isinstance(eval_obj, Eval) for eval_obj in evals)


def test_get_default_evals() -> None:
    """Tests getting default evals."""
    # Act
    evals = get_default_evals()

    # Assert
    assert evals is not None
    assert len(evals) > 0
    assert all(isinstance(eval_obj, Eval) for eval_obj in evals)


def test_eval_class_initialization() -> None:
    """Tests initializing the Eval class."""
    # Act
    eval_obj = Eval(
        name="test_eval",
        criteria="Test evaluation criteria",
        query_identifier="Query",
        context_identifier="Context",
        response_identifier="Response",
    )

    # Assert
    assert eval_obj is not None
    assert eval_obj.name == "test_eval"
    assert eval_obj.criteria == "Test evaluation criteria"
    assert eval_obj.query_identifier == "Query"
    assert eval_obj.context_identifier == "Context"
    assert eval_obj.response_identifier == "Response"


def test_eval_class_with_defaults() -> None:
    """Tests initializing the Eval class with default values."""
    # Act
    eval_obj = Eval(
        name="test_eval",
        criteria="Test evaluation criteria",
    )

    # Assert
    assert eval_obj is not None
    assert eval_obj.name == "test_eval"
    assert eval_obj.criteria == "Test evaluation criteria"
    assert eval_obj.query_identifier is None
    assert eval_obj.context_identifier is None
    assert eval_obj.response_identifier is None


def test_evaluation_metric_type() -> None:
    """Tests the EvaluationMetric type."""
    # Act
    metric: EvaluationMetric = {
        "score": 0.85,
    }

    # Assert
    assert metric is not None
    assert "score" in metric
    assert metric["score"] == 0.85

    # Act - with log
    metric_with_log: EvaluationMetric = {
        "score": 0.75,
        "log": {"explanation": "This is a test explanation"},
    }

    # Assert
    assert metric_with_log is not None
    assert "score" in metric_with_log
    assert metric_with_log["score"] == 0.75
    assert "log" in metric_with_log
    assert metric_with_log["log"] == {"explanation": "This is a test explanation"}

    # Act - with None score
    metric_with_none_score: EvaluationMetric = {
        "score": None,
    }

    # Assert
    assert metric_with_none_score is not None
    assert "score" in metric_with_none_score
    assert metric_with_none_score["score"] is None


def test_trustworthy_rag_response_type() -> None:
    """Tests the TrustworthyRAGResponse type."""
    # Act
    response: TrustworthyRAGResponse = {
        "response": "This is a test response",
    }

    # Assert
    assert response is not None
    assert "response" in response
    assert response["response"] == "This is a test response"

    # Act - with evaluation metrics
    response_with_metrics: TrustworthyRAGResponse = {
        "response": "This is a test response",
        "context_informativeness": {"score": 0.9},
        "response_helpfulness": {"score": 0.85},
    }

    # Assert
    assert response_with_metrics is not None
    assert "response" in response_with_metrics
    assert response_with_metrics["response"] == "This is a test response"
    assert "context_informativeness" in response_with_metrics
    assert "response_helpfulness" in response_with_metrics
    assert response_with_metrics["context_informativeness"]["score"] == 0.9
    assert response_with_metrics["response_helpfulness"]["score"] == 0.85

    # Act - with None response
    response_with_none: TrustworthyRAGResponse = {
        "response": None,
    }

    # Assert
    assert response_with_none is not None
    assert "response" in response_with_none
    assert response_with_none["response"] is None


def test_trustworthy_rag_score_type() -> None:
    """Tests the TrustworthyRAGScore type."""
    # Act
    score: TrustworthyRAGScore = {
        "context_informativeness": {"score": 0.9},
        "response_helpfulness": {"score": 0.85},
    }

    # Assert
    assert score is not None
    assert "context_informativeness" in score
    assert "response_helpfulness" in score
    assert score["context_informativeness"]["score"] == 0.9
    assert score["response_helpfulness"]["score"] == 0.85

    # Act - with logs
    score_with_logs: TrustworthyRAGScore = {
        "context_informativeness": {
            "score": 0.9,
            "log": {"explanation": "The context is very informative"},
        },
        "response_helpfulness": {
            "score": 0.85,
            "log": {"explanation": "The response is helpful"},
        },
    }

    # Assert
    assert score_with_logs is not None
    assert "context_informativeness" in score_with_logs
    assert "response_helpfulness" in score_with_logs
    assert score_with_logs["context_informativeness"]["score"] == 0.9
    assert score_with_logs["context_informativeness"]["log"]["explanation"] == "The context is very informative"
    assert score_with_logs["response_helpfulness"]["score"] == 0.85
    assert score_with_logs["response_helpfulness"]["log"]["explanation"] == "The response is helpful"

    # Act - with empty scores (an empty dict is not a valid TrustworthyRAGScore)
    score_with_empty_scores: TrustworthyRAGScore = {}

    # Assert
    assert score_with_empty_scores is not None
    assert len(score_with_empty_scores) == 0


def test_generate_with_query_and_context(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests generating a response with query and context."""
    # Act
    response = trustworthy_rag.generate(
        query=test_query,
        context=test_context,
    )

    # Assert
    assert response is not None
    assert is_trustworthy_rag_response(response)
    assert "response" in response
    assert response["response"] is not None


def test_generate_with_prompt(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests generating a response with a prompt."""
    # Act
    response = trustworthy_rag.generate(
        query=test_query,
        context=test_context,
        prompt=test_prompt,
    )

    # Assert
    assert response is not None
    assert is_trustworthy_rag_response(response)
    assert "response" in response
    assert response["response"] is not None


def test_generate_with_custom_form_prompt(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests generating a response with a custom form_prompt function."""
    # Arrange
    def custom_form_prompt(query, context):
        # Example of including a system prompt in the custom form_prompt
        system_prompt = "You are a helpful assistant that provides accurate information based on the context."

        prompt = f"{system_prompt}\n\n"
        prompt += f"CUSTOM PROMPT FORMAT\n\nQUESTION: {query}\n\nINFORMATION: {context}\n\n"
        prompt += "ANSWER:"
        return prompt

    # Act
    response = trustworthy_rag.generate(
        query=test_query,
        context=test_context,
        form_prompt=custom_form_prompt,
    )

    # Assert
    assert response is not None
    assert is_trustworthy_rag_response(response)
    assert "response" in response
    assert response["response"] is not None


def test_generate_missing_required_params(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests generating a response with missing required parameters."""
    # Act & Assert - Missing context
    with pytest.raises(ValidationError):
        trustworthy_rag.generate(query=test_query, context=None)

    # Act & Assert - Missing query
    with pytest.raises(ValidationError):
        trustworthy_rag.generate(query=None, context=test_context)

    # Act & Assert - Both query and context are None
    with pytest.raises(ValidationError):
        trustworthy_rag.generate(query=None, context=None)


def test_score_with_query_context_response(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests scoring a response with a query, context, and response."""
    # Act
    score = trustworthy_rag.score(
        query=test_query,
        context=test_context,
        response=test_response,
    )

    # Assert
    assert score is not None
    assert is_trustworthy_rag_score(score)
    # Check that there is at least one evaluation metric
    assert len(score) > 0
    # Check that each key is an evaluation metric with a score
    for metric_name, metric_data in score.items():
        assert "score" in metric_data


def test_score_with_prompt_and_response(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests scoring a response with a prompt and response."""
    # Act
    score = trustworthy_rag.score(
        prompt=test_prompt,
        response=test_response,
        query=test_query,
        context=test_context,
    )

    # Assert
    assert score is not None
    assert is_trustworthy_rag_score(score)
    # Check that there is at least one evaluation metric
    assert len(score) > 0
    # Check that each key is an evaluation metric with a score
    for metric_name, metric_data in score.items():
        assert "score" in metric_data


def test_score_with_custom_form_prompt(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests scoring a response with a custom form_prompt function."""
    # Arrange
    def custom_form_prompt(query, context):
        # Example of including a system prompt in the custom form_prompt
        system_prompt = "You are a helpful assistant that provides accurate information based on the context."

        prompt = f"{system_prompt}\n\n"
        prompt += f"CUSTOM PROMPT FORMAT\n\nQUESTION: {query}\n\nINFORMATION: {context}\n\n"
        prompt += "ANSWER:"
        return prompt

    # Act
    score = trustworthy_rag.score(
        response=test_response,
        query=test_query,
        context=test_context,
        form_prompt=custom_form_prompt,
    )

    # Assert
    assert score is not None
    assert is_trustworthy_rag_score(score)
    # Check that there is at least one evaluation metric
    assert len(score) > 0
    # Check that each key is an evaluation metric with a score
    for metric_name, metric_data in score.items():
        assert "score" in metric_data


def test_validate_inputs_method(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests the _validate_inputs method."""
    # Act & Assert - Valid inputs for generate
    is_valid, error_message = trustworthy_rag._validate_inputs(
        query=test_query,
        context=test_context,
    )
    assert is_valid
    assert error_message is None

    # Act & Assert - Valid inputs for score
    is_valid, error_message = trustworthy_rag._validate_inputs(
        query=test_query,
        context=test_context,
        response=test_response,
    )
    assert is_valid
    assert error_message is None

    # Act & Assert - Missing query
    is_valid, error_message = trustworthy_rag._validate_inputs(
        context=test_context,
    )
    assert not is_valid
    assert error_message is not None

    # Act & Assert - Missing context
    is_valid, error_message = trustworthy_rag._validate_inputs(
        query=test_query,
    )
    assert not is_valid
    assert error_message is not None


def test_default_prompt_formatter(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests the default prompt formatter."""
    # Act
    prompt = trustworthy_rag._default_prompt_formatter(
        query=test_query,
        context=test_context,
    )

    # Assert
    assert prompt is not None
    assert isinstance(prompt, str)
    assert test_query in prompt
    assert test_context in prompt


def test_score_missing_required_params(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests scoring a response with missing required parameters."""
    # Act & Assert - Missing response
    with pytest.raises(ValidationError):
        trustworthy_rag.score(
            response=None,
            query=test_query,
            context=test_context,
        )

    # Act & Assert - Missing both prompt and query/context
    with pytest.raises(ValidationError):
        trustworthy_rag.score(
            response=test_response,
            query=None,
            context=None,
        )

    # Act & Assert - Missing context
    with pytest.raises(ValidationError):
        trustworthy_rag.score(
            response=test_response,
            query=test_query,
            context=None
        )

    # Act & Assert - Missing query
    with pytest.raises(ValidationError):
        trustworthy_rag.score(
            response=test_response,
            query=None,
            context=test_context
        )


def test_generate_with_prompt_and_form_prompt(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests that providing both prompt and form_prompt raises a ValidationError."""
    # Arrange
    def custom_form_prompt(query, context):
        return f"Custom prompt with query: {query} and context: {context}"

    # Act & Assert
    with pytest.raises(ValidationError):
        trustworthy_rag.generate(
            query=test_query,
            context=test_context,
            prompt=test_prompt,
            form_prompt=custom_form_prompt,
        )


def test_score_with_prompt_and_form_prompt(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests scoring a response with both prompt and form_prompt, which should raise an error."""
    # Arrange
    def custom_form_prompt(query, context):
        return f"Question: {query}\nContext: {context}"

    # Act & Assert
    with pytest.raises(ValidationError):
        trustworthy_rag.score(
            response=test_response,
            query=test_query,
            context=test_context,
            prompt=test_prompt,
            form_prompt=custom_form_prompt,
        )


def test_init_with_unsupported_quality_preset(trustworthy_rag_api_key: str) -> None:
    """Tests initializing TrustworthyRAG with an unsupported quality preset."""
    # Act & Assert
    with pytest.raises(ValidationError):
        TrustworthyRAG(
            quality_preset="unsupported_preset",
            api_key=trustworthy_rag_api_key,
        )


# Batch functionality tests

def test_generate_with_batch_query_and_context(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests generating responses with batch queries and contexts."""
    # Arrange
    batch_queries = [test_query, test_query + " And what is the Eiffel Tower?"]
    batch_contexts = [test_context, test_context + " The Eiffel Tower is a famous landmark in Paris."]

    # Act
    responses = trustworthy_rag.generate(
        query=batch_queries,
        context=batch_contexts,
    )

    # Assert
    assert responses is not None
    assert isinstance(responses, list)
    assert len(responses) == 2

    for response in responses:
        assert is_trustworthy_rag_response(response)
        assert "response" in response
        assert response["response"] is not None


def test_generate_with_batch_prompt(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests generating responses with batch prompts."""
    # Arrange
    batch_prompts = [
        f"Question: {test_query}\nContext: {test_context}\nAnswer:",
        f"Question: {test_query} And what is the Eiffel Tower?\nContext: {test_context} The Eiffel Tower is a famous landmark in Paris.\nAnswer:"
    ]

    # Act
    responses = trustworthy_rag.generate(
        query=test_query,
        context=test_context,
        prompt=batch_prompts,
    )

    # Assert
    assert responses is not None
    assert isinstance(responses, list)
    assert len(responses) == 2

    for response in responses:
        assert is_trustworthy_rag_response(response)
        assert "response" in response
        assert response["response"] is not None


def test_generate_with_mixed_batch_inputs(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests generating responses with mixed batch inputs (batch queries, single context)."""
    # Arrange
    batch_queries = [test_query, test_query + " And what is the Eiffel Tower?"]

    # Act
    responses = trustworthy_rag.generate(
        query=batch_queries,
        context=test_context,
    )

    # Assert
    assert responses is not None
    assert isinstance(responses, list)
    assert len(responses) == 2

    for response in responses:
        assert is_trustworthy_rag_response(response)
        assert "response" in response
        assert response["response"] is not None


def test_generate_with_batch_custom_form_prompt(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests generating responses with batch inputs and a custom form_prompt function."""
    # Arrange
    batch_queries = [test_query, test_query + " And what is the Eiffel Tower?"]
    batch_contexts = [test_context, test_context + " The Eiffel Tower is a famous landmark in Paris."]

    def custom_form_prompt(query, context):
        return f"CUSTOM FORMAT\nQ: {query}\nC: {context}\nA:"

    # Act
    responses = trustworthy_rag.generate(
        query=batch_queries,
        context=batch_contexts,
        form_prompt=custom_form_prompt,
    )

    # Assert
    assert responses is not None
    assert isinstance(responses, list)
    assert len(responses) == 2

    for response in responses:
        assert is_trustworthy_rag_response(response)
        assert "response" in response
        assert response["response"] is not None


def test_score_with_batch_inputs(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests scoring responses with batch inputs."""
    # Arrange
    batch_queries = [test_query, test_query + " And what is the Eiffel Tower?"]
    batch_contexts = [test_context, test_context + " The Eiffel Tower is a famous landmark in Paris."]
    batch_responses = [test_response, "The capital of France is Paris. The Eiffel Tower is a famous landmark in Paris."]

    # Act
    scores = trustworthy_rag.score(
        query=batch_queries,
        context=batch_contexts,
        response=batch_responses,
    )

    # Assert
    assert scores is not None
    assert isinstance(scores, list)
    assert len(scores) == 2

    for score in scores:
        assert is_trustworthy_rag_score(score)


def test_score_with_batch_prompt(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests scoring responses with batch prompts."""
    # Arrange
    batch_prompts = [
        f"Question: {test_query}\nContext: {test_context}\nAnswer:",
        f"Question: {test_query} And what is the Eiffel Tower?\nContext: {test_context} The Eiffel Tower is a famous landmark in Paris.\nAnswer:"
    ]
    batch_responses = [test_response, "The capital of France is Paris. The Eiffel Tower is a famous landmark in Paris."]

    # Act
    scores = trustworthy_rag.score(
        prompt=batch_prompts,
        response=batch_responses,
        query=test_query,
        context=test_context,
    )

    # Assert
    assert scores is not None
    assert isinstance(scores, list)
    assert len(scores) == 2

    for score in scores:
        assert is_trustworthy_rag_score(score)


def test_score_with_mixed_batch_inputs(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests scoring responses with mixed batch inputs (batch responses, single query and context)."""
    # Arrange
    batch_responses = [test_response, "The capital of France is Paris. The Eiffel Tower is a famous landmark in Paris."]

    # Act
    scores = trustworthy_rag.score(
        query=test_query,
        context=test_context,
        response=batch_responses,
    )

    # Assert
    assert scores is not None
    assert isinstance(scores, list)
    assert len(scores) == 2

    for score in scores:
        assert is_trustworthy_rag_score(score)


def test_try_generate_with_batch_inputs(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests try_generate with batch inputs."""
    # Arrange
    batch_queries = [test_query, test_query + " And what is the Eiffel Tower?"]
    batch_contexts = [test_context, test_context + " The Eiffel Tower is a famous landmark in Paris."]

    # Act
    responses = trustworthy_rag.try_generate(
        query=batch_queries,
        context=batch_contexts,
    )

    # Assert
    assert responses is not None
    assert isinstance(responses, list)
    assert len(responses) == 2

    for response in responses:
        assert is_trustworthy_rag_response(response)
        assert "response" in response or "error" in response  # May contain response or error


def test_try_generate_with_custom_form_prompt(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests try_generate with batch inputs and a custom form_prompt function."""
    # Arrange
    batch_queries = [test_query, test_query + " And what is the Eiffel Tower?"]
    batch_contexts = [test_context, test_context + " The Eiffel Tower is a famous landmark in Paris."]

    def custom_form_prompt(query, context):
        return f"CUSTOM FORMAT\nQ: {query}\nC: {context}\nA:"

    # Act
    responses = trustworthy_rag.try_generate(
        query=batch_queries,
        context=batch_contexts,
        form_prompt=custom_form_prompt,
    )

    # Assert
    assert responses is not None
    assert isinstance(responses, list)
    assert len(responses) == 2

    for response in responses:
        assert is_trustworthy_rag_response(response)
        assert "response" in response or "error" in response  # May contain response or error


def test_try_score_with_batch_inputs(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests try_score with batch inputs."""
    # Arrange
    batch_queries = [test_query, test_query + " And what is the Eiffel Tower?"]
    batch_contexts = [test_context, test_context + " The Eiffel Tower is a famous landmark in Paris."]
    batch_responses = [test_response, "The capital of France is Paris. The Eiffel Tower is a famous landmark in Paris."]

    # Act
    scores = trustworthy_rag.try_score(
        query=batch_queries,
        context=batch_contexts,
        response=batch_responses,
    )

    # Assert
    assert scores is not None
    assert isinstance(scores, list)
    assert len(scores) == 2

    for score in scores:
        assert is_trustworthy_rag_score(score)

def test_try_score_with_custom_form_prompt(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests try_score with batch inputs and a custom form_prompt function."""
    # Arrange
    batch_queries = [test_query, test_query + " And what is the Eiffel Tower?"]
    batch_contexts = [test_context, test_context + " The Eiffel Tower is a famous landmark in Paris."]
    batch_responses = [test_response, "The capital of France is Paris. The Eiffel Tower is a famous landmark in Paris."]

    def custom_form_prompt(query, context):
        return f"CUSTOM FORMAT\nQ: {query}\nC: {context}\nA:"

    # Act
    scores = trustworthy_rag.try_score(
        query=batch_queries,
        context=batch_contexts,
        response=batch_responses,
        form_prompt=custom_form_prompt,
    )

    # Assert
    assert scores is not None
    assert isinstance(scores, list)
    assert len(scores) == 2

    for score in scores:
        assert is_trustworthy_rag_score(score)

def test_batch_validation_errors(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests validation errors with batch inputs of different lengths."""
    # Arrange
    batch_queries = [test_query, test_query + " And what is the Eiffel Tower?"]
    batch_contexts = [test_context]  # Only one context for two queries

    # Act & Assert - generate
    with pytest.raises(ValidationError):
        trustworthy_rag.generate(
            query=batch_queries,
            context=batch_contexts,
        )

    # Arrange for score
    batch_responses = [test_response]  # Only one response for two queries

    # Act & Assert - score
    with pytest.raises(ValidationError):
        trustworthy_rag.score(
            query=batch_queries,
            context=batch_contexts,
            response=batch_responses,
        )

    # Act & Assert - try_generate
    with pytest.raises(ValidationError):
        trustworthy_rag.try_generate(
            query=batch_queries,
            context=batch_contexts,
        )

    # Act & Assert - try_score
    with pytest.raises(ValidationError):
        trustworthy_rag.try_score(
            query=batch_queries,
            context=batch_contexts,
            response=batch_responses,
        )


@pytest.fixture(autouse=True)
def reset_trustworthy_rag(trustworthy_rag: TrustworthyRAG) -> None:
    """Reset the TrustworthyRAG instance after each test."""
    return
    # No reset needed for TrustworthyRAG as it doesn't have state that needs to be reset


def test_batch_processing_with_mixed_input_types(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests batch processing with mixed input types (sequence and single values)."""
    # Arrange
    batch_queries = [test_query, test_query + " And what is the Eiffel Tower?"]
    single_context = test_context

    # Act - Generate with sequence of queries and single context
    responses = trustworthy_rag.generate(
        query=batch_queries,
        context=single_context,
    )

    # Assert
    assert responses is not None
    assert isinstance(responses, list)
    assert len(responses) == 2

    for response in responses:
        assert is_trustworthy_rag_response(response)
        assert "response" in response
        assert response["response"] is not None

    # Act - Score with single query and sequence of contexts
    batch_contexts = [test_context, test_context + " The Eiffel Tower is a famous landmark in Paris."]
    batch_responses = [r["response"] for r in responses]

    scores = trustworthy_rag.score(
        query=test_query,
        context=batch_contexts,
        response=batch_responses,
    )

    # Assert
    assert scores is not None
    assert isinstance(scores, list)
    assert len(scores) == 2

    for score in scores:
        assert is_trustworthy_rag_score(score)


def test_batch_processing_with_prompt_parameter(trustworthy_rag: TrustworthyRAG) -> None:
    """Tests batch processing with prompt parameter."""
    # Arrange
    batch_prompts = [
        f"Question: {test_query}\nContext: {test_context}\nAnswer:",
        f"Question: {test_query} And what is the Eiffel Tower?\nContext: {test_context} The Eiffel Tower is a famous landmark in Paris.\nAnswer:"
    ]

    # Act - Generate with batch prompts
    responses = trustworthy_rag.generate(
        query=test_query,  # These are required but will be ignored when prompt is provided
        context=test_context,
        prompt=batch_prompts,
    )

    # Assert
    assert responses is not None
    assert isinstance(responses, list)
    assert len(responses) == 2

    for response in responses:
        assert is_trustworthy_rag_response(response)
        assert "response" in response
        assert response["response"] is not None

    # Act - Score with batch prompts
    batch_responses = [r["response"] for r in responses]

    scores = trustworthy_rag.score(
        query=test_query,  # These are required but will be ignored when prompt is provided
        context=test_context,
        response=batch_responses,
        prompt=batch_prompts,
    )

    # Assert
    assert scores is not None
    assert isinstance(scores, list)
    assert len(scores) == 2

    for score in scores:
        assert is_trustworthy_rag_score(score)

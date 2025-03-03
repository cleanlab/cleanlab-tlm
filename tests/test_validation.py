from typing import Any

import numpy as np
import pytest

from cleanlab_tlm.errors import TlmBadRequestError, ValidationError
from cleanlab_tlm.internal.constants import (
    _VALID_TLM_TASKS,
    TLM_TASK_SUPPORTING_CONSTRAIN_OUTPUTS,
)
from cleanlab_tlm.tlm import TLM, TLMOptions
from tests.conftest import make_text_unique
from tests.constants import (
    CHARACTERS_PER_TOKEN,
    MAX_COMBINED_LENGTH_TOKENS,
    MAX_PROMPT_LENGTH_TOKENS,
    MAX_RESPONSE_LENGTH_TOKENS,
    TEST_PROMPT,
    TEST_PROMPT_BATCH,
    TEST_RESPONSE,
)
from tests.test_get_trustworthiness_score import is_tlm_score_response_with_error
from tests.test_prompt import is_tlm_response_with_error

np.random.seed(0)
test_prompt = make_text_unique(TEST_PROMPT)
test_prompt_batch = [make_text_unique(prompt) for prompt in TEST_PROMPT_BATCH]


def assert_prompt_too_long_error(response: Any, index: int) -> None:
    assert is_tlm_response_with_error(response)
    assert response["log"]["error"]["message"].startswith(f"Error executing query at index {index}:")
    assert "Prompt length exceeds maximum length of 70000 tokens" in response["log"]["error"]["message"]
    assert response["log"]["error"]["retryable"] is False


def assert_prompt_too_long_error_score(response: Any, index: int) -> None:
    assert is_tlm_score_response_with_error(response)
    assert response["log"]["error"]["message"].startswith(f"Error executing query at index {index}:")
    assert "Prompt length exceeds maximum length of 70000 tokens" in response["log"]["error"]["message"]
    assert response["log"]["error"]["retryable"] is False


def assert_response_too_long_error_score(response: Any, index: int) -> None:
    assert is_tlm_score_response_with_error(response)
    assert response["log"]["error"]["message"].startswith(f"Error executing query at index {index}:")
    assert "Response length exceeds maximum length of 15000 tokens" in response["log"]["error"]["message"]
    assert response["log"]["error"]["retryable"] is False


def assert_prompt_and_response_combined_too_long_error_score(response: Any, index: int) -> None:
    assert is_tlm_score_response_with_error(response)
    assert response["log"]["error"]["message"].startswith(f"Error executing query at index {index}:")
    assert (
        "Prompt and response combined length exceeds maximum combined length of 70000 tokens"
        in response["log"]["error"]["message"]
    )
    assert response["log"]["error"]["retryable"] is False


def test_prompt_unsupported_kwargs(tlm: TLM) -> None:
    """Tests that validation error is raised when unsupported keyword arguments are passed to prompt."""
    with pytest.raises(ValidationError) as exc_info:
        tlm.prompt(
            "test prompt",
            constrain_outputss=[["test constrain outputs"]],
        )

    assert str(exc_info.value).startswith("Unsupported keyword arguments: {'constrain_outputss'}")


def test_prompt_constrain_outputs_wrong_type_single_prompt(tlm: TLM) -> None:
    """Tests that validation error is raised when constrain_outputs is not a list of strings when prompt is a string."""
    with pytest.raises(ValidationError) as exc_info:
        tlm.prompt(
            "test prompt",
            constrain_outputs="test constrain outputs",
        )

    assert str(exc_info.value).startswith("constrain_outputs must be a list of strings")


def test_prompt_constrain_outputs_wrong_length(tlm: TLM) -> None:
    """Tests that validation error is raised when constrain_outputs length does not match prompt length."""
    with pytest.raises(ValidationError) as exc_info:
        tlm.prompt(
            ["test prompt"],
            constrain_outputs=[["test constrain outputs"], ["test constrain outputs"]],
        )

    assert str(exc_info.value).startswith("constrain_outputs must have same length as prompt")


def test_prompt_not_providing_constrain_outputs_for_classification_task(
    tlm_api_key: str,
) -> None:
    """Tests that validation error is raised when constrain_outputs is not provided for classification tasks."""
    tlm_classification = TLM(api_key=tlm_api_key, task="classification")
    with pytest.raises(ValidationError) as exc_info:
        tlm_classification.prompt(
            "test prompt",
        )

    assert str(exc_info.value).startswith("constrain_outputs must be provided for classification tasks")


@pytest.mark.parametrize("task", _VALID_TLM_TASKS - TLM_TASK_SUPPORTING_CONSTRAIN_OUTPUTS)
def test_prompt_providing_constrain_outputs_for_non_classification_task(
    tlm_api_key: str,
    task: str,
) -> None:
    """Tests that validation error is raised when constrain_outputs is provided for non-classification tasks."""
    tlm = TLM(api_key=tlm_api_key, task=task)
    with pytest.raises(ValidationError) as exc_info:
        tlm.prompt(
            "test prompt",
            constrain_outputs="test constrain outputs",
        )

    assert str(exc_info.value).startswith("constrain_outputs is only supported for classification tasks")


def test_scoring_constrain_outputs_wrong_type_single_prompt(tlm: TLM) -> None:
    """Tests that validation error is raised when constrain_outputs is not a list of strings when prompt is a string."""
    with pytest.raises(ValidationError) as exc_info:
        tlm.get_trustworthiness_score(
            "test prompt",
            "test response",
            constrain_outputs="test constrain outputs",
        )

    assert str(exc_info.value).startswith("constrain_outputs must be a list of strings")


def test_scoring_constrain_outputs_wrong_length(tlm: TLM) -> None:
    """Tests that validation error is raised when constrain_outputs length does not match prompt length."""
    with pytest.raises(ValidationError) as exc_info:
        tlm.get_trustworthiness_score(
            ["test prompt"],
            ["test response"],
            constrain_outputs=[["test constrain outputs"], ["test constrain outputs"]],
        )

    assert str(exc_info.value).startswith("constrain_outputs must have same length as prompt")


def test_scoring_not_providing_constrain_outputs_for_classification_task(
    tlm_api_key: str,
) -> None:
    """Tests that validation error is raised when constrain_outputs is not provided for classification tasks."""
    tlm_classification = TLM(api_key=tlm_api_key, task="classification")
    with pytest.raises(ValidationError) as exc_info:
        tlm_classification.get_trustworthiness_score(
            "test prompt",
            "test response",
        )

    assert str(exc_info.value).startswith("constrain_outputs must be provided for classification tasks")


@pytest.mark.parametrize("task", _VALID_TLM_TASKS - TLM_TASK_SUPPORTING_CONSTRAIN_OUTPUTS)
def test_scoring_providing_constrain_outputs_for_non_classification_task(
    tlm_api_key: str,
    task: str,
) -> None:
    """Tests that validation error is raised when constrain_outputs is provided for non-classification tasks."""
    tlm = TLM(api_key=tlm_api_key, task=task)
    with pytest.raises(ValidationError) as exc_info:
        tlm.get_trustworthiness_score(
            "test prompt",
            "test response",
            constrain_outputs="test constrain outputs",
        )

    assert str(exc_info.value).startswith("constrain_outputs is only supported for classification tasks")


def test_scoring_response_not_in_constrain_outputs(tlm: TLM) -> None:
    """Tests that validation error is raised when response is not in constrain_outputs."""
    with pytest.raises(ValidationError) as exc_info:
        tlm.get_trustworthiness_score(
            "test prompt",
            "test response",
            constrain_outputs=["test constrain outputs"],
        )

    assert str(exc_info.value).startswith(
        "Response 'test response' must be one of the constraint outputs: ['test constrain outputs']"
    )


def test_scoring_response_not_in_constrain_outputs_batch(tlm: TLM) -> None:
    """Tests that validation error is raised when response is not in constrain_outputs."""
    with pytest.raises(ValidationError) as exc_info:
        tlm.get_trustworthiness_score(
            ["test prompt1", "test prompt2"],
            ["test response1", "test response2"],
            constrain_outputs=[["test response1"], ["test constrain outputs"]],
        )

    assert str(exc_info.value).startswith(
        "Response 'test response2' at index 1 must be one of the constraint outputs: ['test constrain outputs']"
    )


def test_prompt_too_long_exception_single_prompt(tlm: TLM) -> None:
    """Tests that bad request error is raised when prompt is too long when calling tlm.prompt with a single prompt."""
    with pytest.raises(TlmBadRequestError) as exc_info:
        tlm.prompt(
            "a" * (MAX_PROMPT_LENGTH_TOKENS + 1) * CHARACTERS_PER_TOKEN,
        )

    assert exc_info.value.message.startswith("Prompt length exceeds")
    assert exc_info.value.retryable is False


@pytest.mark.parametrize("num_prompts", [1, 2, 5])
def test_prompt_too_long_exception_batch_prompt(tlm: TLM, num_prompts: int) -> None:
    """Tests that bad request error is raised when prompt is too long when calling tlm.prompt with a batch of prompts.

    Error message should indicate which the batch index for which the prompt is too long.
    """
    # create batch of prompts with one prompt that is too long
    prompts = [test_prompt] * num_prompts
    prompt_too_long_index = np.random.randint(0, num_prompts)
    prompts[prompt_too_long_index] = "a" * (MAX_PROMPT_LENGTH_TOKENS + 1) * CHARACTERS_PER_TOKEN

    with pytest.raises(TlmBadRequestError) as exc_info:
        tlm.prompt(prompts)

    assert exc_info.value.message.startswith(f"Error executing query at index {prompt_too_long_index}:")
    assert "Prompt length exceeds" in exc_info.value.message
    assert exc_info.value.retryable is False


@pytest.mark.parametrize("num_prompts", [1, 2, 5])
def test_prompt_too_long_exception_try_prompt(tlm: TLM, num_prompts: int) -> None:
    """Tests that None is returned when prompt is too long when calling tlm.try_prompt with a batch of prompts."""
    # create batch of prompts with one prompt that is too long
    prompts = [test_prompt] * num_prompts
    prompt_too_long_index = np.random.randint(0, num_prompts)
    prompts[prompt_too_long_index] = "a" * (MAX_PROMPT_LENGTH_TOKENS + 1) * CHARACTERS_PER_TOKEN

    tlm_responses = tlm.try_prompt(
        prompts,
    )

    assert_prompt_too_long_error(tlm_responses[prompt_too_long_index], prompt_too_long_index)


def test_response_too_long_exception_single_score(tlm: TLM) -> None:
    """Tests that bad request error is raised when response is too long when calling tlm.get_trustworthiness_score with a single prompt."""
    with pytest.raises(TlmBadRequestError) as exc_info:
        tlm.get_trustworthiness_score(
            "a",
            "a" * (MAX_RESPONSE_LENGTH_TOKENS + 1) * CHARACTERS_PER_TOKEN,
        )

    assert exc_info.value.message.startswith("Response length exceeds")
    assert exc_info.value.retryable is False


@pytest.mark.parametrize("num_prompts", [1, 2, 5])
def test_response_too_long_exception_batch_score(tlm: TLM, num_prompts: int) -> None:
    """Tests that bad request error is raised when prompt is too long when calling tlm.get_trustworthiness_score with a batch of prompts.

    Error message should indicate which the batch index for which the prompt is too long.
    """
    # create batch of prompts with one prompt that is too long
    prompts = [test_prompt] * num_prompts
    responses = [TEST_RESPONSE] * num_prompts
    response_too_long_index = np.random.randint(0, num_prompts)
    responses[response_too_long_index] = "a" * (MAX_RESPONSE_LENGTH_TOKENS + 1) * CHARACTERS_PER_TOKEN

    with pytest.raises(TlmBadRequestError) as exc_info:
        tlm.get_trustworthiness_score(
            prompts,
            responses,
        )

    assert exc_info.value.message.startswith(f"Error executing query at index {response_too_long_index}:")
    assert "Response length exceeds" in exc_info.value.message
    assert exc_info.value.retryable is False


@pytest.mark.parametrize("num_prompts", [1, 2, 5])
def test_response_too_long_exception_try_score(tlm: TLM, num_prompts: int) -> None:
    """Tests that None is returned when prompt is too long when calling tlm.try_get_trustworthiness_score with a batch of prompts."""
    # create batch of prompts with one prompt that is too long
    prompts = [test_prompt] * num_prompts
    responses = [TEST_RESPONSE] * num_prompts
    response_too_long_index = np.random.randint(0, num_prompts)
    responses[response_too_long_index] = "a" * (MAX_RESPONSE_LENGTH_TOKENS + 1) * CHARACTERS_PER_TOKEN

    tlm_responses = tlm.try_get_trustworthiness_score(
        prompts,
        responses,
    )

    assert_response_too_long_error_score(tlm_responses[response_too_long_index], response_too_long_index)


def test_prompt_too_long_exception_single_score(tlm: TLM) -> None:
    """Tests that bad request error is raised when prompt is too long when calling tlm.get_trustworthiness_score with a single prompt."""
    with pytest.raises(TlmBadRequestError) as exc_info:
        tlm.get_trustworthiness_score(
            "a" * (MAX_PROMPT_LENGTH_TOKENS + 1) * CHARACTERS_PER_TOKEN,
            "a",
        )

    assert exc_info.value.message.startswith("Prompt length exceeds")
    assert exc_info.value.retryable is False


@pytest.mark.parametrize("num_prompts", [1, 2, 5])
def test_prompt_too_long_exception_batch_score(tlm: TLM, num_prompts: int) -> None:
    """Tests that bad request error is raised when prompt is too long when calling tlm.get_trustworthiness_score with a batch of prompts.

    Error message should indicate which the batch index for which the prompt is too long.
    """
    # create batch of prompts with one prompt that is too long
    prompts = [test_prompt] * num_prompts
    responses = [TEST_RESPONSE] * num_prompts
    prompt_too_long_index = np.random.randint(0, num_prompts)
    prompts[prompt_too_long_index] = "a" * (MAX_PROMPT_LENGTH_TOKENS + 1) * CHARACTERS_PER_TOKEN

    with pytest.raises(TlmBadRequestError) as exc_info:
        tlm.get_trustworthiness_score(
            prompts,
            responses,
        )

    assert exc_info.value.message.startswith(f"Error executing query at index {prompt_too_long_index}:")
    assert "Prompt length exceeds" in exc_info.value.message
    assert exc_info.value.retryable is False


@pytest.mark.parametrize("num_prompts", [1, 2, 5])
def test_prompt_too_long_exception_try_score(tlm: TLM, num_prompts: int) -> None:
    """Tests that None is returned when prompt is too long when calling tlm.try_get_trustworthiness_score with a batch of prompts."""
    # create batch of prompts with one prompt that is too long
    prompts = [test_prompt] * num_prompts
    responses = [TEST_RESPONSE] * num_prompts
    prompt_too_long_index = np.random.randint(0, num_prompts)
    prompts[prompt_too_long_index] = "a" * (MAX_PROMPT_LENGTH_TOKENS + 1) * CHARACTERS_PER_TOKEN

    tlm_responses = tlm.try_get_trustworthiness_score(
        prompts,
        responses,
    )

    assert_prompt_too_long_error_score(tlm_responses[prompt_too_long_index], prompt_too_long_index)


def test_combined_too_long_exception_single_score(tlm: TLM) -> None:
    """Tests that bad request error is raised when prompt + response combined length is too long when calling tlm.get_trustworthiness_score with a single prompt."""
    max_prompt_length = MAX_COMBINED_LENGTH_TOKENS - MAX_RESPONSE_LENGTH_TOKENS + 1

    with pytest.raises(TlmBadRequestError) as exc_info:
        tlm.get_trustworthiness_score(
            "a" * max_prompt_length * CHARACTERS_PER_TOKEN,
            "a" * MAX_RESPONSE_LENGTH_TOKENS * CHARACTERS_PER_TOKEN,
        )

    assert exc_info.value.message.startswith("Prompt and response combined length exceeds")
    assert exc_info.value.retryable is False


@pytest.mark.parametrize("num_prompts", [1, 2, 5])
def test_prompt_and_response_combined_too_long_exception_batch_score(tlm: TLM, num_prompts: int) -> None:
    """Tests that bad request error is raised when prompt + response combined length is too long when calling tlm.get_trustworthiness_score with a batch of prompts.

    Error message should indicate which the batch index for which the prompt is too long.
    """
    # create batch of prompts with one prompt that is too long
    prompts = [test_prompt] * num_prompts
    responses = [TEST_RESPONSE] * num_prompts
    combined_too_long_index = np.random.randint(0, num_prompts)

    max_prompt_length = MAX_COMBINED_LENGTH_TOKENS - MAX_RESPONSE_LENGTH_TOKENS + 1
    prompts[combined_too_long_index] = "a" * max_prompt_length * CHARACTERS_PER_TOKEN
    responses[combined_too_long_index] = "a" * MAX_RESPONSE_LENGTH_TOKENS * CHARACTERS_PER_TOKEN

    tlm_responses = tlm.try_get_trustworthiness_score(
        prompts,
        responses,
    )

    assert_prompt_and_response_combined_too_long_error_score(
        tlm_responses[combined_too_long_index], combined_too_long_index
    )


@pytest.mark.parametrize("num_prompts", [1, 2, 5])
def test_prompt_and_response_combined_too_long_exception_try_score(tlm: TLM, num_prompts: int) -> None:
    """Tests that appropriate error is returned when prompt + response is too long when calling tlm.try_get_trustworthiness_score with a batch of prompts."""
    # create batch of prompts with one prompt that is too long
    prompts = [test_prompt] * num_prompts
    responses = [TEST_RESPONSE] * num_prompts
    combined_too_long_index = np.random.randint(0, num_prompts)
    max_prompt_length = MAX_COMBINED_LENGTH_TOKENS - MAX_RESPONSE_LENGTH_TOKENS + 1
    prompts[combined_too_long_index] = "a" * max_prompt_length * CHARACTERS_PER_TOKEN
    responses[combined_too_long_index] = "a" * MAX_RESPONSE_LENGTH_TOKENS * CHARACTERS_PER_TOKEN

    tlm_responses = tlm.try_get_trustworthiness_score(
        prompts,
        responses,
    )

    assert_prompt_and_response_combined_too_long_error_score(
        tlm_responses[combined_too_long_index], combined_too_long_index
    )


def test_invalid_option_passed(tlm_api_key: str) -> None:
    """Tests that validation error is thrown when an invalid option is passed to the TLM."""
    invalid_option = "invalid_option"
    with pytest.raises(
        ValidationError,
        match=f"^Invalid keys in options dictionary: {{'{invalid_option}'}}.*",
    ):
        TLM(
            api_key=tlm_api_key,
            options=TLMOptions(invalid_option="invalid_value"),  # type: ignore[typeddict-unknown-key]
        )


def test_max_tokens_invalid_option_passed(tlm_api_key: str) -> None:
    """Tests that validation error is thrown when an invalid max_tokens option value is passed to the TLM."""
    max_tokens = -1
    with pytest.raises(ValidationError, match=f"Invalid value {max_tokens}, max_tokens.*"):
        TLM(api_key=tlm_api_key, options=TLMOptions(max_tokens=max_tokens))


def test_validate_rag_inputs_prompt_and_form_prompt_together() -> None:
    """Tests that ValidationError is raised when both prompt and form_prompt are provided."""
    from cleanlab_tlm.internal.validation import validate_rag_inputs

    def form_prompt_func(query: str, context: str) -> str:
        return f"Query: {query}\nContext: {context}"

    with pytest.raises(ValidationError) as exc_info:
        validate_rag_inputs(
            query="test query",
            context="test context",
            prompt="test prompt",
            form_prompt=form_prompt_func,
            is_generate=True,
        )

    assert "prompt' and 'form_prompt' cannot be provided at the same time" in str(exc_info.value)


def test_validate_rag_inputs_batch_not_supported() -> None:
    """Tests that NotImplementedError is raised when batch inputs are provided."""
    from cleanlab_tlm.internal.validation import validate_rag_inputs

    with pytest.raises(NotImplementedError) as exc_info:
        validate_rag_inputs(query=["query1", "query2"], context="test context", is_generate=True)

    assert "Batch processing is not yet supported" in str(exc_info.value)


def test_validate_rag_inputs_generate_missing_required_params() -> None:
    """Tests that ValidationError is raised when required parameters are missing for generate."""
    from cleanlab_tlm.internal.validation import validate_rag_inputs

    # Missing context
    with pytest.raises(ValidationError) as exc_info:
        validate_rag_inputs(query="test query", context=None, is_generate=True)  # type: ignore

    assert "Both 'query' and 'context' are required parameters" in str(exc_info.value)

    # Missing query
    with pytest.raises(ValidationError) as exc_info:
        validate_rag_inputs(query=None, context="test context", is_generate=True)  # type: ignore

    assert "Both 'query' and 'context' are required parameters" in str(exc_info.value)


def test_validate_rag_inputs_score_missing_required_params() -> None:
    """Tests that ValidationError is raised when required parameters are missing for score."""
    from cleanlab_tlm.internal.validation import validate_rag_inputs

    # Missing response
    with pytest.raises(ValidationError) as exc_info:
        validate_rag_inputs(query="test query", context="test context", is_generate=False)

    assert "'response' is a required parameter" in str(exc_info.value)

    # Missing query and context when prompt is not provided
    with pytest.raises(ValidationError) as exc_info:
        validate_rag_inputs(query=None, context=None, response="test response", is_generate=False)  # type: ignore

    assert "Either 'prompt' or both 'query' and 'context' must be provided" in str(exc_info.value)


def test_validate_rag_inputs_form_prompt_missing_params() -> None:
    """Tests that ValidationError is raised when form_prompt is provided but query or context is missing."""
    from cleanlab_tlm.internal.validation import validate_rag_inputs

    def form_prompt_func(query: str, context: str) -> str:
        return f"Query: {query}\nContext: {context}"

    with pytest.raises(ValidationError) as exc_info:
        validate_rag_inputs(query=None, context=None, form_prompt=form_prompt_func, is_generate=True)  # type: ignore

    # The function first checks for required parameters before checking form_prompt specifics
    assert "Both 'query' and 'context' are required parameters" in str(exc_info.value)


def test_validate_rag_inputs_invalid_param_types() -> None:
    """Tests that ValidationError is raised when parameters have invalid types."""
    from cleanlab_tlm.internal.validation import validate_rag_inputs

    with pytest.raises(ValidationError) as exc_info:
        validate_rag_inputs(
            query=123,  # type: ignore
            context="test context",
            is_generate=True,
        )

    assert "'query' must be a string" in str(exc_info.value)


def test_validate_rag_inputs_with_evals() -> None:
    """Tests that ValidationError is raised when required inputs for evaluations are missing."""
    from cleanlab_tlm.internal.validation import validate_rag_inputs

    class MockEval:
        def __init__(
            self,
            name: str,
            query_identifier: bool = False,
            context_identifier: bool = False,
            response_identifier: bool = False,
        ):
            self.name = name
            self.query_identifier = query_identifier
            self.context_identifier = context_identifier
            self.response_identifier = response_identifier

    # For this test, we need to provide the required parameters first to reach the eval validation
    # Test missing response for eval that requires it (score mode)
    with pytest.raises(ValidationError) as exc_info:
        validate_rag_inputs(
            query="test query",
            context="test context",
            prompt="test prompt",
            response=None,  # Missing response
            evals=[MockEval("test_eval", response_identifier=True)],
            is_generate=False,
        )

    assert "'response' is a required parameter" in str(exc_info.value)

    # Let's test a successful case to ensure the validation passes when all requirements are met
    result = validate_rag_inputs(
        prompt="test prompt",
        query="test query",
        context="test context",
        response="test response",
        evals=[MockEval("test_eval", query_identifier=True, context_identifier=True, response_identifier=True)],
        is_generate=False,
    )

    assert result == "test prompt"


def test_validate_rag_inputs_successful_generate() -> None:
    """Tests that validate_rag_inputs returns the formatted prompt when validation succeeds for generate."""
    from cleanlab_tlm.internal.validation import validate_rag_inputs

    # Test with direct prompt
    result = validate_rag_inputs(query="test query", context="test context", prompt="test prompt", is_generate=True)

    assert result == "test prompt"

    # Test with form_prompt function
    def form_prompt_func(query: str, context: str) -> str:
        return f"Query: {query}\nContext: {context}"

    result = validate_rag_inputs(
        query="test query", context="test context", form_prompt=form_prompt_func, is_generate=True
    )

    assert result == "Query: test query\nContext: test context"


def test_validate_rag_inputs_successful_score() -> None:
    """Tests that validate_rag_inputs returns the formatted prompt when validation succeeds for score."""
    from cleanlab_tlm.internal.validation import validate_rag_inputs

    # Test with direct prompt
    result = validate_rag_inputs(
        query="test query", context="test context", response="test response", prompt="test prompt", is_generate=False
    )

    assert result == "test prompt"

    # Test with form_prompt function
    def form_prompt_func(query: str, context: str) -> str:
        return f"Query: {query}\nContext: {context}"

    result = validate_rag_inputs(
        query="test query",
        context="test context",
        response="test response",
        form_prompt=form_prompt_func,
        is_generate=False,
    )

    assert result == "Query: test query\nContext: test context"


def test_custom_eval_criteria_validation(tlm_api_key: str) -> None:
    """Tests validation of custom_eval_criteria."""
    # Valid custom_eval_criteria
    valid_criteria = [{"name": "test_criteria", "criteria": "This is a test criteria"}]

    # Should work fine
    tlm = TLM(api_key=tlm_api_key, options=TLMOptions(custom_eval_criteria=valid_criteria))
    assert tlm is not None

    # Invalid: not a list
    with pytest.raises(
        ValidationError,
        match="^Invalid type.*custom_eval_criteria must be a list of dictionaries.$",
    ):
        TLM(
            api_key=tlm_api_key,
            options=TLMOptions(custom_eval_criteria="not a list"),  # type: ignore
        )

    # Invalid: item not a dictionary
    with pytest.raises(
        ValidationError,
        match="^Item 0 in custom_eval_criteria is not a dictionary.$",
    ):
        TLM(
            api_key=tlm_api_key,
            options=TLMOptions(custom_eval_criteria=["not a dict"]),  # type: ignore
        )

    # Invalid: missing name
    with pytest.raises(
        ValidationError,
        match="^Missing required key 'name' in custom_eval_criteria item 0.$",
    ):
        TLM(
            api_key=tlm_api_key,
            options=TLMOptions(custom_eval_criteria=[{"criteria": "test"}]),
        )

    # Invalid: missing criteria
    with pytest.raises(
        ValidationError,
        match="^Missing required key 'criteria' in custom_eval_criteria item 0.$",
    ):
        TLM(
            api_key=tlm_api_key,
            options=TLMOptions(custom_eval_criteria=[{"name": "test"}]),
        )

    # Invalid: name not a string
    with pytest.raises(
        ValidationError,
        match="^'name' in custom_eval_criteria item 0 must be a string.$",
    ):
        TLM(
            api_key=tlm_api_key,
            options=TLMOptions(custom_eval_criteria=[{"name": 123, "criteria": "test"}]),
        )

    # Invalid: criteria not a string
    with pytest.raises(
        ValidationError,
        match="^'criteria' in custom_eval_criteria item 0 must be a string.$",
    ):
        TLM(
            api_key=tlm_api_key,
            options=TLMOptions(custom_eval_criteria=[{"name": "test", "criteria": 123}]),
        )


def test_validate_tlm_options_support_custom_eval_criteria() -> None:
    """Tests that validate_tlm_options correctly handles support_custom_eval_criteria parameter."""
    from cleanlab_tlm.internal.validation import validate_tlm_options

    # Valid options with support_custom_eval_criteria=True
    options = {"custom_eval_criteria": [{"name": "test", "criteria": "test criteria"}]}
    validate_tlm_options(options, support_custom_eval_criteria=True)

    # Invalid options with support_custom_eval_criteria=False
    with pytest.raises(
        ValidationError,
        match="^custom_eval_criteria is not supported for this class$",
    ):
        validate_tlm_options(options, support_custom_eval_criteria=False)

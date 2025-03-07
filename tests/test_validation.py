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

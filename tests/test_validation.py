from typing import Any, cast

import numpy as np
import pytest

from cleanlab_tlm.errors import TlmBadRequestError, ValidationError
from cleanlab_tlm.internal.constants import (
    _VALID_TLM_TASKS,
    TLM_TASK_SUPPORTING_CONSTRAIN_OUTPUTS,
)
from cleanlab_tlm.tlm import TLM, TLMOptions, TLMResponse, TLMScore
from cleanlab_tlm.utils.rag import Eval, TrustworthyRAG
from tests.conftest import make_text_unique
from tests.constants import (
    MAX_COMBINED_LENGTH_TOKENS,
    MAX_PROMPT_LENGTH_TOKENS,
    MAX_RESPONSE_LENGTH_TOKENS,
    TEST_PROMPT,
    TEST_PROMPT_BATCH,
    TEST_RESPONSE,
    WORD_THAT_EQUALS_ONE_TOKEN,
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
            WORD_THAT_EQUALS_ONE_TOKEN * (MAX_PROMPT_LENGTH_TOKENS + 1),
        )

    assert exc_info.value.message.startswith("Prompt length exceeds")
    assert exc_info.value.retryable is False


@pytest.mark.parametrize("num_prompts", [1, 2, 5])
def test_prompt_too_long_exception_prompt(tlm: TLM, num_prompts: int) -> None:
    """Tests that None is returned when prompt is too long when calling tlm.prompt with a batch of prompts."""
    # create batch of prompts with one prompt that is too long
    prompts = [test_prompt] * num_prompts
    prompt_too_long_index = np.random.randint(0, num_prompts)
    prompts[prompt_too_long_index] = WORD_THAT_EQUALS_ONE_TOKEN * (MAX_PROMPT_LENGTH_TOKENS + 1)

    tlm_responses = cast(list[TLMResponse], tlm.prompt(prompts))

    assert_prompt_too_long_error(tlm_responses[prompt_too_long_index], prompt_too_long_index)


def test_response_too_long_exception_single_score(tlm: TLM) -> None:
    """Tests that bad request error is raised when response is too long when calling tlm.get_trustworthiness_score with a single prompt."""
    with pytest.raises(TlmBadRequestError) as exc_info:
        tlm.get_trustworthiness_score(
            WORD_THAT_EQUALS_ONE_TOKEN,
            WORD_THAT_EQUALS_ONE_TOKEN * (MAX_RESPONSE_LENGTH_TOKENS + 1),
        )

    assert exc_info.value.message.startswith("Response length exceeds")
    assert exc_info.value.retryable is False


@pytest.mark.parametrize("num_prompts", [1, 2, 5])
def test_response_too_long_exception_score(tlm: TLM, num_prompts: int) -> None:
    """Tests that None is returned when prompt is too long when calling tlm.get_trustworthiness_score with a batch of prompts."""
    # create batch of prompts with one prompt that is too long
    prompts = [test_prompt] * num_prompts
    responses = [TEST_RESPONSE] * num_prompts
    response_too_long_index = np.random.randint(0, num_prompts)
    responses[response_too_long_index] = WORD_THAT_EQUALS_ONE_TOKEN * (MAX_RESPONSE_LENGTH_TOKENS + 1)

    tlm_responses = cast(list[TLMScore], tlm.get_trustworthiness_score(prompts, responses))

    assert_response_too_long_error_score(tlm_responses[response_too_long_index], response_too_long_index)


def test_prompt_too_long_exception_single_score(tlm: TLM) -> None:
    """Tests that bad request error is raised when prompt is too long when calling tlm.get_trustworthiness_score with a single prompt."""
    with pytest.raises(TlmBadRequestError) as exc_info:
        tlm.get_trustworthiness_score(
            WORD_THAT_EQUALS_ONE_TOKEN * (MAX_PROMPT_LENGTH_TOKENS + 1),
            WORD_THAT_EQUALS_ONE_TOKEN,
        )

    assert exc_info.value.message.startswith("Prompt length exceeds")
    assert exc_info.value.retryable is False


@pytest.mark.parametrize("num_prompts", [1, 2, 5])
def test_prompt_too_long_exception_score(tlm: TLM, num_prompts: int) -> None:
    """Tests that None is returned when prompt is too long when calling tlm.get_trustworthiness_score with a batch of prompts."""
    # create batch of prompts with one prompt that is too long
    prompts = [test_prompt] * num_prompts
    responses = [TEST_RESPONSE] * num_prompts
    prompt_too_long_index = np.random.randint(0, num_prompts)
    prompts[prompt_too_long_index] = WORD_THAT_EQUALS_ONE_TOKEN * (MAX_PROMPT_LENGTH_TOKENS + 1)

    tlm_responses = cast(list[TLMScore], tlm.get_trustworthiness_score(prompts, responses))

    assert_prompt_too_long_error_score(tlm_responses[prompt_too_long_index], prompt_too_long_index)


def test_combined_too_long_exception_single_score(tlm: TLM) -> None:
    """Tests that bad request error is raised when prompt + response combined length is too long when calling tlm.get_trustworthiness_score with a single prompt."""
    max_prompt_length = MAX_COMBINED_LENGTH_TOKENS - MAX_RESPONSE_LENGTH_TOKENS + 1

    with pytest.raises(TlmBadRequestError) as exc_info:
        tlm.get_trustworthiness_score(
            WORD_THAT_EQUALS_ONE_TOKEN * max_prompt_length,
            WORD_THAT_EQUALS_ONE_TOKEN * MAX_RESPONSE_LENGTH_TOKENS,
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
    prompts[combined_too_long_index] = WORD_THAT_EQUALS_ONE_TOKEN * max_prompt_length
    responses[combined_too_long_index] = WORD_THAT_EQUALS_ONE_TOKEN * MAX_RESPONSE_LENGTH_TOKENS

    tlm_responses = cast(list[TLMScore], tlm.get_trustworthiness_score(prompts, responses))

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

    assert "Both 'query' and 'context' are required parameters" in str(exc_info.value)


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

    # Test invalid query type
    with pytest.raises(ValidationError) as exc_info:
        validate_rag_inputs(
            query=123,  # type: ignore
            context="test context",
            is_generate=True,
        )
    assert "'query' must be either a string or a sequence of strings, not <class 'int'>" in str(exc_info.value)

    # Test invalid context type
    with pytest.raises(ValidationError) as exc_info:
        validate_rag_inputs(
            query="test query",
            context=456,  # type: ignore
            is_generate=True,
        )
    assert "'context' must be either a string or a sequence of strings, not <class 'int'>" in str(exc_info.value)

    # Test invalid response type
    with pytest.raises(ValidationError) as exc_info:
        validate_rag_inputs(
            query="test query",
            context="test context",
            response=789,  # type: ignore
            is_generate=False,
        )
    assert "'response' must be either a string or a sequence of strings, not <class 'int'>" in str(exc_info.value)

    # Test invalid prompt type
    with pytest.raises(ValidationError) as exc_info:
        validate_rag_inputs(
            query="test query",
            context="test context",
            prompt=True,  # type: ignore
            is_generate=True,
        )
    assert "'prompt' must be either a string or a sequence of strings, not <class 'bool'>" in str(exc_info.value)

    # Test sequence with non-string items
    with pytest.raises(ValidationError) as exc_info:
        validate_rag_inputs(
            query=["valid", 123, "also valid"],  # type: ignore
            context=["test context1", "test context2", "test context3"],
            is_generate=True,
        )
    assert "All items in 'query' must be of type string when providing a sequence" in str(exc_info.value)

    # Test mismatched length of query and context sequences with a proper form_prompt function
    def valid_form_prompt(q: str, c: str) -> str:
        return f"{q} {c}"

    with pytest.raises(ValidationError) as exc_info:
        validate_rag_inputs(
            query=["q1", "q2", "q3"],
            context=["c1", "c2"],
            form_prompt=valid_form_prompt,
            is_generate=True,
        )
    assert (
        "Input lists have different lengths: query: 3, context: 2. All input lists must have the same length."
        in str(exc_info.value)
    )


def test_validate_proper_evals_input(tlm_api_key: str) -> None:
    evals = [
        Eval(
            name="test_eval",
            criteria="This is a test criteria",
            query_identifier="query",
            context_identifier="context",
            response_identifier="response",
        )
    ]

    # test the expected case will work
    tlm_rag = TrustworthyRAG(api_key=tlm_api_key, evals=evals)
    assert tlm_rag is not None

    # test passing a list of list of evals
    with pytest.raises(ValidationError) as exc_info:
        tlm_rag = TrustworthyRAG(api_key=tlm_api_key, evals=[evals])  # type: ignore
    assert "'evals' must be a list of Eval objects" in str(exc_info.value)

    # test passing a list of dicts
    with pytest.raises(ValidationError) as exc_info:
        tlm_rag = TrustworthyRAG(api_key=tlm_api_key, evals=[{"test_eval": "This is a test criteria"}])  # type: ignore
    assert "'evals' must be a list of Eval objects" in str(exc_info.value)

    # test passing a string
    with pytest.raises(ValidationError) as exc_info:
        tlm_rag = TrustworthyRAG(api_key=tlm_api_key, evals="This is a test criteria")  # type: ignore
    assert "'evals' must be a list of Eval objects" in str(exc_info.value)


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
        match="^Missing required keys {'name'} in custom_eval_criteria item 0.$",
    ):
        TLM(
            api_key=tlm_api_key,
            options=TLMOptions(custom_eval_criteria=[{"criteria": "test"}]),
        )

    # Invalid: missing criteria
    with pytest.raises(
        ValidationError,
        match="^Missing required keys {'criteria'} in custom_eval_criteria item 0.$",
    ):
        TLM(
            api_key=tlm_api_key,
            options=TLMOptions(custom_eval_criteria=[{"name": "test"}]),
        )

    # Invalid: extra keys
    with pytest.raises(
        ValidationError,
        match="^Invalid keys {'extra'} found in custom_eval_criteria item 0. Supported keys are: ({'name', 'criteria'}|{'criteria', 'name'}).$",
    ):
        TLM(
            api_key=tlm_api_key,
            options=TLMOptions(custom_eval_criteria=[{"name": "test", "criteria": "test", "extra": "extra"}]),
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


def test_validate_rag_inputs_mixed_string_and_sequence() -> None:
    """Tests that validate_rag_inputs rejects mixed inputs where some are strings and others are sequences."""
    from cleanlab_tlm.internal.validation import validate_rag_inputs

    # Test string query with sequence context (for generate)
    with pytest.raises(ValidationError) as exc_info:
        validate_rag_inputs(
            query="test query",
            context=["context 1", "context 2"],
            is_generate=True,
        )

    assert "Inconsistent input formats" in str(exc_info.value)
    assert "query is a string while other inputs are lists" in str(exc_info.value)

    # Test with prompt as a sequence but query and context as strings (for score)
    with pytest.raises(ValidationError) as exc_info:
        validate_rag_inputs(
            prompt=["prompt 1", "prompt 2", "prompt 3"],
            query="test query",
            context="test context",
            response="test response",
            is_generate=False,
        )

    assert "Inconsistent input formats" in str(exc_info.value)

    # Test with response as a sequence but query and context as strings (for score)
    with pytest.raises(ValidationError) as exc_info:
        validate_rag_inputs(
            prompt="test prompt",
            query="test query",
            context="test context",
            response=["response 1", "response 2"],
            is_generate=False,
        )

    assert "Inconsistent input formats" in str(exc_info.value)

    # Test the specific case provided by the user
    with pytest.raises(ValidationError) as exc_info:
        validate_rag_inputs(
            prompt=["sample"],
            response=["sample"],
            query="sample",
            context="sample",
            is_generate=False,
        )

    assert "Inconsistent input formats" in str(exc_info.value)


def test_validate_rag_inputs_list_length_mismatch() -> None:
    """Tests that validate_rag_inputs rejects lists with different lengths."""
    from cleanlab_tlm.internal.validation import validate_rag_inputs

    # Test lists with different lengths (for generate)
    with pytest.raises(ValidationError) as exc_info:
        validate_rag_inputs(
            query=["query 1", "query 2", "query 3"],
            context=["context 1", "context 2"],
            is_generate=True,
        )

    assert "Input lists have different lengths" in str(exc_info.value)

    # Test lists with different lengths (for score)
    with pytest.raises(ValidationError) as exc_info:
        validate_rag_inputs(
            query=["query 1", "query 2"],
            context=["context 1", "context 2"],
            response=["response 1", "response 2", "response 3"],
            is_generate=False,
        )

    assert "Input lists have different lengths" in str(exc_info.value)

    # Test lists with different lengths including prompt (for score)
    with pytest.raises(ValidationError) as exc_info:
        validate_rag_inputs(
            prompt=["prompt 1", "prompt 2", "prompt 3"],
            query=["query 1", "query 2"],
            context=["context 1", "context 2"],
            response=["response 1", "response 2"],
            is_generate=False,
        )

    assert "Input lists have different lengths" in str(exc_info.value)


def test_validate_rag_inputs_matching_lists() -> None:
    """Tests that validate_rag_inputs accepts lists with matching lengths."""
    from cleanlab_tlm.internal.validation import validate_rag_inputs

    list_length = 2

    # Test with matching list lengths and form_prompt (for score)
    result = validate_rag_inputs(
        query=["query 1", "query 2"],
        context=["context 1", "context 2"],
        response=["response 1", "response 2"],
        form_prompt=lambda q, c: f"Q: {q} C: {c}",
        is_generate=False,
    )

    # Should get a list of formatted prompts
    assert isinstance(result, list)
    assert len(result) == list_length
    assert result[0] == "Q: query 1 C: context 1"
    assert result[1] == "Q: query 2 C: context 2"

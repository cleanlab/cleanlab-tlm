from collections.abc import Generator
from typing import Any

import pytest

from cleanlab_tlm.errors import APITimeoutError
from cleanlab_tlm.tlm import TLM
from tests.conftest import make_text_unique
from tests.constants import (
    TEST_PROMPT,
    TEST_PROMPT_BATCH,
    TEST_RESPONSE,
    TEST_RESPONSE_BATCH,
)

test_prompt = make_text_unique(TEST_PROMPT)
test_prompt_batch = [make_text_unique(prompt) for prompt in TEST_PROMPT_BATCH]


def is_trustworthiness_score_json_format(response: Any, allow_null_trustworthiness_score: bool = False) -> bool:
    """Returns True if the response is a trustworthiness score in JSON format with valid range."""
    return (
        isinstance(response, dict)
        and "trustworthiness_score" in response
        and (
            (allow_null_trustworthiness_score and response["trustworthiness_score"] is None)
            or (
                isinstance(response["trustworthiness_score"], float) and 0.0 <= response["trustworthiness_score"] <= 1.0
            )
        )
    )


def is_tlm_score_response_with_error(response: Any) -> bool:
    """Returns True if the response matches the expected TLMScore with error format."""
    return (
        isinstance(response, dict)
        and "trustworthiness_score" in response
        and response["trustworthiness_score"] is None
        and (
            isinstance(response["log"], dict)
            and "error" in response["log"]
            and isinstance(response["log"]["error"], dict)
            and "message" in response["log"]["error"]
            and "retryable" in response["log"]["error"]
            and isinstance(response["log"]["error"]["retryable"], bool)
        )
    )


def test_single_get_trustworthiness_score(tlm: TLM) -> None:
    """Tests running a single get_trustworthiness_score in the TLM.

    Expected:
    - TLM should return a single response
    - Response should be non-None
    - No exceptions are raised
    """
    # act -- run a single get_trustworthiness_score
    response = tlm.get_trustworthiness_score(test_prompt, TEST_RESPONSE)

    # assert
    # - response is not None
    # - a single response of type TLMResponse is returned
    # - no exceptions are raised (implicit)
    assert response is not None
    assert is_trustworthiness_score_json_format(response)


def test_single_get_trustworthiness_score_constrain_outputs(tlm: TLM) -> None:
    """Tests running a single get_trustworthiness_score in the TLM with constrain_outputs.
    Expected:
    - TLM should return a single response
    - Response should be non-None
    - No exceptions are raised
    """
    # act -- run a single get_trustworthiness_score
    response = tlm.get_trustworthiness_score(test_prompt, TEST_RESPONSE, constrain_outputs=[TEST_RESPONSE])

    # assert
    # - response is not None
    # - a single response of type TLMResponse is returned
    # - no exceptions are raised (implicit)
    assert response is not None
    assert is_trustworthiness_score_json_format(response)


def test_single_get_trustworthiness_score_force_timeouts(tlm: TLM) -> None:
    """Tests running a single get_trustworthiness_score in the TLM, forcing timeouts.

    Sets timeout to 0.0001 seconds, which should force a timeout for all get_trustworthiness_scores.
    This should result in a timeout error being thrown

    Expected:
    - TLM should raise a timeout error
    """
    # arrange -- override timeout
    tlm._timeout = 0.0001

    # assert -- timeout is thrown
    with pytest.raises(APITimeoutError):
        # act -- run a single get_trustworthiness_score
        tlm.get_trustworthiness_score(
            test_prompt,
            TEST_RESPONSE,
        )


def test_batch_get_trustworthiness_score_constrain_outputs(tlm: TLM) -> None:
    """Tests running a batch get_trustworthiness_score in the TLM with constrain_outputs.
    Expected:
    - TLM should return a list of responses
    - Responses should be non-None
    - No exceptions are raised
    - Each response should be of type TLMResponse
    """
    # act -- run a batch get_trustworthiness_score
    response = tlm.get_trustworthiness_score(
        test_prompt_batch, TEST_RESPONSE_BATCH, constrain_outputs=TEST_RESPONSE_BATCH
    )

    # assert
    # - response is not None
    # - a list of responses of type TLMResponse is returned
    # - no exceptions are raised (implicit)
    assert response is not None
    assert isinstance(response, list)
    assert all(is_trustworthiness_score_json_format(r) for r in response)


def test_batch_get_trustworthiness_score(tlm: TLM) -> None:
    """Tests running a batch get_trustworthiness_score in the TLM.

    Expected:
    - TLM should return a list of responses
    - Responses will be of type TLMResponse
    - No exceptions are raised
    """
    # act -- run a batch get_trustworthiness_score
    response = tlm.get_trustworthiness_score(
        test_prompt_batch,
        TEST_RESPONSE_BATCH,
    )

    # assert
    # - response is not None
    # - a list of responses of type TLMResponse or None is returned
    # - no exceptions are raised (implicit)
    assert response is not None
    assert isinstance(response, list)
    assert all(is_trustworthiness_score_json_format(r) for r in response)


def test_batch_get_trustworthiness_score_force_timeouts(tlm: TLM) -> None:
    """Tests running a batch get_trustworthiness_score in the TLM, forcing timeouts.

    Sets timeout to 0.0001 seconds, which should force a timeout for all get_trustworthiness_scores.
    This should result in TLMResponse with error messages and retryability information for all get_trustworthiness_scores.

    Expected:
    - TLM should return a list of responses
    - Responses will be of type TLMResponse
    - No exceptions are raised
    """
    # arrange -- override timeout
    tlm._timeout = 0.0001

    # act -- run a batch get_trustworthiness_score
    response = tlm.get_trustworthiness_score(
        test_prompt_batch,
        TEST_RESPONSE_BATCH,
    )

    # assert
    # - response is not None
    # - all responses timed out and are None
    # - no exceptions are raised (implicit)
    assert response is not None
    assert isinstance(response, list)
    assert all(is_tlm_score_response_with_error(r) for r in response)


def test_try_get_trustworthiness_score(tlm: TLM) -> None:
    # act -- run a batch get_trustworthiness_score
    with pytest.warns(DeprecationWarning) as warning:
        response = tlm.try_get_trustworthiness_score(
            test_prompt_batch,
            TEST_RESPONSE_BATCH,
        )

    # assert
    # - response is not None
    # - a list of responses of type TLMResponse or None is returned
    # - no exceptions are raised (implicit)
    assert response is not None
    assert isinstance(response, list)
    assert all(is_trustworthiness_score_json_format(r) for r in response)
    assert "Deprecated method." in str(warning[0].message)


@pytest.fixture(autouse=True)
def reset_tlm(tlm: TLM) -> Generator[None, None, None]:
    original_timeout = tlm._timeout
    yield
    tlm._timeout = original_timeout

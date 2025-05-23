from collections.abc import Generator
from typing import Any

import pytest

from cleanlab_tlm.errors import APITimeoutError
from cleanlab_tlm.tlm import TLM
from tests.conftest import make_text_unique
from tests.constants import TEST_PROMPT, TEST_PROMPT_BATCH

test_prompt = make_text_unique(TEST_PROMPT)
test_prompt_batch = [make_text_unique(prompt) for prompt in TEST_PROMPT_BATCH]


def is_tlm_response(
    response: Any,
    allow_none_response: bool = False,
    allow_null_trustworthiness_score: bool = False,
) -> bool:
    """Returns True if the response is a TLMResponse.

    Args:
        allow_none_response: If True, allows the response to be None
        allow_null_trustworthiness_score: If True, allows the trustworthiness_score to be None
            (only allowed for base preset for models with no perplexity score)
    """
    # check if response is allowed to be none
    if response is None:
        return allow_none_response

    if isinstance(response, dict) and "response" in response and "trustworthiness_score" in response:
        trustworthiness_score = response["trustworthiness_score"]

        # check if trustworthiness score is allowed to be none
        if trustworthiness_score is None:
            return allow_null_trustworthiness_score

        return isinstance(trustworthiness_score, float) and 0.0 <= trustworthiness_score <= 1.0

    return False


def is_tlm_response_with_error(response: Any) -> bool:
    """Returns True if the response is a TLMResponse with an error."""
    return (
        isinstance(response, dict)
        and "response" in response
        and response["response"] is None
        and "trustworthiness_score" in response
        and response["trustworthiness_score"] is None
        and "log" in response
        and "error" in response["log"]
        and "message" in response["log"]["error"]
        and "retryable" in response["log"]["error"]
    )


def test_single_prompt(tlm: TLM) -> None:
    """Tests running a single prompt in the TLM.

    Expected:
    - TLM should return a single response
    - Response should be non-None
    - No exceptions are raised
    """

    # act -- run a single prompt
    response = tlm.prompt(test_prompt)

    # assert
    # - response is not None
    # - a single response of type TLMResponse is returned
    # - no exceptions are raised (implicit)
    assert response is not None
    assert is_tlm_response(response)


def test_single_prompt_constrain_outputs(tlm: TLM) -> None:
    """Tests running a single prompt in the TLM with constrain_outputs.

    Expected:
    - TLM should return a single response
    - Response should be non-None
    - No exceptions are raised
    """

    # act -- run a single prompt
    response = tlm.prompt(test_prompt, constrain_outputs=["test"])

    # assert
    # - response is not None
    # - a single response of type TLMResponse is returned
    # - no exceptions are raised (implicit)
    assert response is not None
    assert is_tlm_response(response)
    assert isinstance(response, dict)
    assert response["response"] == "test"


def test_single_prompt_force_timeouts(tlm: TLM) -> None:
    """Tests running a single prompt in the TLM, forcing timeouts.

    Sets timeout to 0.0001 seconds, which should force a timeout for all prompts.
    This should result in a timeout error being thrown

    Expected:
    - TLM should raise a timeout error
    """
    # arrange -- override timeout
    tlm._timeout = 0.0001

    # assert -- timeout is thrown
    with pytest.raises(APITimeoutError):
        # act -- run a single prompt
        tlm.prompt(test_prompt)


def test_batch_prompt(tlm: TLM) -> None:
    """Tests running a batch prompt in the TLM.

    Expected:
    - TLM should return a list of responses
    - Responses can be None or of type TLMResponse
    - No exceptions are raised
    """
    # act -- run a batch prompt
    response = tlm.prompt(test_prompt_batch)

    # assert
    # - response is not None
    # - a list of responses of type TLMResponse is returned
    # - no exceptions are raised (implicit)
    assert response is not None
    assert isinstance(response, list)
    assert all(is_tlm_response(r, allow_none_response=True) for r in response)


def test_batch_prompt_constrain_outputs(tlm: TLM) -> None:
    """Tests running a batch prompt in the TLM with constrain_outputs.

    Expected:
    - TLM should return a list of responses
    - Responses can be None or of type TLMResponse
    - No exceptions are raised
    """
    # act -- run a batch prompt
    response = tlm.prompt(test_prompt_batch, constrain_outputs=[["test"], ["test"]])

    # assert
    # - response is not None
    # - a list of responses of type TLMResponse is returned
    # - no exceptions are raised (implicit)
    assert response is not None
    assert isinstance(response, list)
    assert all(is_tlm_response(r, allow_none_response=True) for r in response)
    assert all(r["response"] == "test" for r in response)


def test_batch_prompt_force_timeouts(tlm: TLM) -> None:
    """Tests running a batch prompt in the TLM, forcing timeouts.

    Sets timeout to 0.0001 seconds, which should force a timeout for all prompts.
    This should result in None responses for all prompts.

    Expected:
    - TLM should return a list of responses
    - Responses can be None or of type TLMResponse
    - No exceptions are raised
    """
    # arrange -- override timeout
    tlm._timeout = 0.0001

    # act -- run a batch prompt
    response = tlm.prompt(test_prompt_batch)

    # assert
    # - response is not None
    # - all responses timed out and are None
    # - no exceptions are raised (implicit)
    assert response is not None
    assert isinstance(response, list)
    assert all(is_tlm_response_with_error(r) for r in response)


def test_try_prompt(tlm: TLM) -> None:
    # act -- run a batch get_trustworthiness_score
    with pytest.warns(DeprecationWarning) as warning:
        response = tlm.try_prompt(
            test_prompt_batch,
        )

    # assert
    # - response is not None
    # - a list of responses of type TLMResponse or None is returned
    # - no exceptions are raised (implicit)
    assert response is not None
    assert isinstance(response, list)
    assert all(is_tlm_response(r, allow_none_response=True) for r in response)
    assert "Deprecated method." in str(warning[0].message)


@pytest.fixture(autouse=True)
def reset_tlm(tlm: TLM) -> Generator[None, None, None]:
    original_timeout = tlm._timeout
    yield
    tlm._timeout = original_timeout

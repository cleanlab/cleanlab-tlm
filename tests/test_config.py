import pytest

from cleanlab_tlm.errors import TlmBadRequestError
from cleanlab_tlm.tlm import TLM
from cleanlab_tlm.utils.config import get_default_context_limit, get_default_model, get_default_quality_preset, get_default_max_tokens
from tests.constants import WORD_THAT_EQUALS_ONE_TOKEN

tlm_with_default_setting = TLM()


def test_get_default_model(tlm: TLM) -> None:
    assert tlm.get_model_name() == get_default_model()


def test_get_default_quality_preset(tlm: TLM) -> None:
    assert get_default_quality_preset() == tlm._quality_preset


def test_prompt_too_long_exception_single_prompt(tlm: TLM) -> None:
    """Tests that bad request error is raised when prompt is too long when calling tlm.prompt with a single prompt."""
    with pytest.raises(TlmBadRequestError) as exc_info:
        tlm.prompt(WORD_THAT_EQUALS_ONE_TOKEN * (get_default_context_limit() + 1))

    assert exc_info.value.message.startswith("Prompt with token count")
    assert exc_info.value.retryable is False


def test_prompt_within_context_limit_returns_response(tlm: TLM) -> None:
    """Tests that no error is raised when prompt length is within limit."""
    response = tlm.prompt(WORD_THAT_EQUALS_ONE_TOKEN * (get_default_context_limit() - 1000))

    assert isinstance(response, dict)
    assert "response" in response
    assert isinstance(response["response"], str)
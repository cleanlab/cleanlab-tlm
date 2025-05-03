import pytest
import tiktoken

from cleanlab_tlm.errors import TlmBadRequestError
from cleanlab_tlm.tlm import TLM
from cleanlab_tlm.utils.config import (
    get_default_context_limit,
    get_default_max_tokens,
    get_default_model,
    get_default_quality_preset,
)
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

    assert exc_info.value.message.startswith("Prompt length exceeds")
    assert exc_info.value.retryable is False


def test_prompt_within_context_limit_returns_response(tlm: TLM) -> None:
    """Tests that no error is raised when prompt length is within limit."""
    response = tlm.prompt(WORD_THAT_EQUALS_ONE_TOKEN * (get_default_context_limit() - 1000))

    assert isinstance(response, dict)
    assert "response" in response
    assert isinstance(response["response"], str)


def test_response_within_max_tokens() -> None:
    """Tests that response is within max tokens limit."""
    tlm_base = TLM(quality_preset="base")
    prompt = "write a 100 page book about computer science. make sure it is extremely long and comprehensive."

    result = tlm_base.prompt(prompt)
    assert isinstance(result, dict)
    response = result["response"]
    assert isinstance(response, str)

    try:
        enc = tiktoken.encoding_for_model(get_default_model())
    except KeyError:
        enc = tiktoken.encoding_for_model("gpt-4o")
    tokens_in_response = len(enc.encode(response))
    assert tokens_in_response <= get_default_max_tokens()

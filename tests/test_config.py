import pytest

from cleanlab_tlm.tlm import TLM
from cleanlab_tlm.errors import TlmBadRequestError, ValidationError
from tests.constants import (
    CHARACTERS_PER_TOKEN,
)
from cleanlab_tlm.utils.config import (
    get_default_model,
    get_default_quality_preset,
    get_default_context_limit,
    get_default_max_tokens,
)

tlm_with_default_setting = TLM()

def assert_get_default_model_matches_tlm_default_setting(tlm: TLM) -> None:
    assert tlm.get_model_name() == get_default_model()

def test_prompt_too_long_exception_single_prompt(tlm: TLM) -> None:
    """Tests that bad request error is raised when prompt is too long when calling tlm.prompt with a single prompt."""
    with pytest.raises(TlmBadRequestError) as exc_info:
        tlm.prompt(     
            "a" * (get_default_context_limit() + 1) * CHARACTERS_PER_TOKEN,
        )

    assert exc_info.value.message.startswith("Prompt length exceeds")
    assert exc_info.value.retryable is False


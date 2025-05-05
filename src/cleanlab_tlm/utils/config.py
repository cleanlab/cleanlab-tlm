from cleanlab_tlm.internal.constants import (
    _DEFAULT_TLM_MAX_TOKENS,
    _DEFAULT_TLM_QUALITY_PRESET,
    _TLM_DEFAULT_CONTEXT_LIMIT,
    _TLM_DEFAULT_MODEL,
)


def get_default_model() -> str:
    """
    Get the default model name for TLM.

    Returns:
        str: The default model name for TLM.
    """
    return _TLM_DEFAULT_MODEL


def get_default_quality_preset() -> str:
    """
    Get the default quality preset for TLM.

    Returns:
        str: The default quality preset for TLM.
    """
    return _DEFAULT_TLM_QUALITY_PRESET


def get_default_context_limit() -> int:
    """
    Get the default context limit for TLM.

    Returns:
        int: The default context limit for TLM.
    """
    return _TLM_DEFAULT_CONTEXT_LIMIT


def get_default_max_tokens() -> int:
    """
    Get the default maximum output tokens allowed.

    Returns:
        int: The default maximum output tokens.
    """
    return _DEFAULT_TLM_MAX_TOKENS

from cleanlab_tlm.internal.constants import (
    _TLM_DEFAULT_MODEL,
    _DEFAULT_TLM_QUALITY_PRESET,
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

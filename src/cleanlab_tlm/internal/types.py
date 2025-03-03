from enum import Enum
from typing import Any, Literal

TLMQualityPreset = Literal["best", "high", "medium", "low", "base"]


class Task(str, Enum):
    """Enum for TLM task types."""

    DEFAULT = "default"
    CLASSIFICATION = "classification"
    CODE_GENERATION = "code_generation"


JSONDict = dict[str, Any]


# Simple type alias for use in exception_handling.py
# This avoids circular imports while still providing type information
TLMResult = dict[str, Any]

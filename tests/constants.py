from cleanlab_tlm.internal.constants import _VALID_TLM_MODELS

# Test TLM
TEST_PROMPT: str = "What is the capital of France?"
TEST_RESPONSE: str = "Paris"
TEST_PROMPT_BATCH: list[str] = [
    "What is the capital of France?",
    "What is the capital of Ukraine?",
]
TEST_RESPONSE_BATCH: list[str] = ["Paris", "Kyiv"]

# Test validation tests for TLM
MAX_PROMPT_LENGTH_TOKENS: int = 70_000
MAX_RESPONSE_LENGTH_TOKENS: int = 15_000
MAX_COMBINED_LENGTH_TOKENS: int = 70_000

CHARACTERS_PER_TOKEN: int = 4

# Property tests for TLM
excluded_tlm_models: list[str] = [
    "claude-3.5-haiku",
    "claude-3-sonnet",
    "claude-3.5-sonnet",
    "claude-3.5-sonnet-v2",
    "o1-preview",
    "o1",
    "o1-mini",
    "o3-mini",
    "nova-lite",
    "nova-pro",
    "gpt-4",
]
VALID_TLM_MODELS: list[str] = [model for model in _VALID_TLM_MODELS if model not in excluded_tlm_models]
MODELS_WITH_NO_PERPLEXITY_SCORE: list[str] = [
    "claude-3-haiku",
    "claude-3.5-haiku",
    "claude-3-sonnet",
    "claude-3.5-sonnet",
    "claude-3.5-sonnet-v2",
    "o1-preview",
    "o1",
    "o1-mini",
    "o3-mini",
    "nova-micro",
    "nova-lite",
    "nova-pro",
]

from cleanlab_tlm.internal.constants import _VALID_TLM_MODELS

# Test TLM
TEST_PROMPT: str = "What is the capital of France?"
TEST_RESPONSE: str = "Paris"
TEST_PROMPT_BATCH: list[str] = [
    "What is the capital of France?",
    "What is the capital of Ukraine?",
]
TEST_RESPONSE_BATCH: list[str] = ["Paris", "Kyiv"]
TEST_CONSTRAIN_OUTPUTS_BINARY = ["Paris", "Kyiv"]
TEST_CONSTRAIN_OUTPUTS = ["Paris", "Kyiv", "London", "New York"]

# Test validation tests for TLM
MAX_PROMPT_LENGTH_TOKENS: int = 70_000
MAX_RESPONSE_LENGTH_TOKENS: int = 15_000
MAX_COMBINED_LENGTH_TOKENS: int = 70_000

CHARACTERS_PER_TOKEN: int = 4
# 4 character (3 character + 1 space) = 1 token
WORD_THAT_EQUALS_ONE_TOKEN = "orb "  # noqa: S105

# Property tests for TLM
excluded_tlm_models: list[str] = [
    "claude-3.5-haiku",
    "claude-3-sonnet",
    "claude-3.5-sonnet",
    "claude-3.5-sonnet-v2",
    "claude-3.7-sonnet",
    "o1-preview",
    "o1",
    "o1-mini",
    "o3",
    "o3-mini",
    "o4-mini",
    "nova-lite",
    "nova-pro",
    "gpt-4",
    "gpt-4.1",
    "gpt-4.5-preview",
]
VALID_TLM_MODELS: list[str] = [model for model in _VALID_TLM_MODELS if model not in excluded_tlm_models]
MODELS_WITH_NO_PERPLEXITY_SCORE: list[str] = [
    "claude-3-haiku",
    "claude-3.5-haiku",
    "claude-3-sonnet",
    "claude-3.5-sonnet",
    "claude-3.5-sonnet-v2",
    "claude-3.7-sonnet",
    "o1-preview",
    "o1",
    "o1-mini",
    "o3",
    "o3-mini",
    "o4-mini",
    "gpt-4.5-preview",
    "nova-micro",
    "nova-lite",
    "nova-pro",
]

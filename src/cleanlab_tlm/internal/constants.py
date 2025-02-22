# TLM constants
# prepend constants with _ so that they don't show up in help.cleanlab.ai docs
_VALID_TLM_QUALITY_PRESETS: list[str] = ["best", "high", "medium", "low", "base"]
_VALID_TLM_MODELS: list[str] = [
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4o",
    "gpt-4o-mini",
    "o1-preview",
    "o1",
    "o1-mini",
    "o3-mini",
    "claude-3-haiku",
    "claude-3.5-haiku",
    "claude-3-sonnet",
    "claude-3.5-sonnet",
    "claude-3.5-sonnet-v2",
    "nova-micro",
    "nova-lite",
    "nova-pro",
]
_TLM_DEFAULT_MODEL: str = "gpt-4o-mini"
_TLM_MAX_RETRIES: int = 3  # TODO: finalize this number
_TLM_MAX_TOKEN_RANGE: dict[str, tuple[int, int]] = {  # model: (min, max)
    "default": (64, 4096),
    "claude-3-haiku": (64, 512),
    "claude-3.5-haiku": (64, 512),
    "claude-3-sonnet": (64, 512),
    "claude-3.5-sonnet": (64, 512),
    "nova-micro": (64, 512),
}
TLM_CONSTRAIN_OUTPUTS_KEY: str = "constrain_outputs"
TLM_NUM_CANDIDATE_RESPONSES_RANGE: tuple[int, int] = (1, 20)  # (min, max)
TLM_NUM_CONSISTENCY_SAMPLES_RANGE: tuple[int, int] = (0, 20)  # (min, max)
TLM_SIMILARITY_MEASURES: set[str] = {"semantic", "string"}
TLM_REASONING_EFFORT_VALUES: set[str] = {"none", "low", "medium", "high"}
TLM_VALID_LOG_OPTIONS: set[str] = {"perplexity", "explanation"}
TLM_VALID_GET_TRUSTWORTHINESS_SCORE_KWARGS: set[str] = {
    "perplexity",
    TLM_CONSTRAIN_OUTPUTS_KEY,
}
TLM_VALID_PROMPT_KWARGS: set[str] = {TLM_CONSTRAIN_OUTPUTS_KEY}
TLM_MODELS_NOT_SUPPORTING_EXPLANATION: set[str] = {
    "o1-mini",
    "o1-preview",
    "o1",
    "o3-mini",
}
VALID_RESPONSE_OPTIONS: set[str] = {"max_tokens"}
INVALID_SCORE_OPTIONS: set[str] = {"num_candidate_responses"}

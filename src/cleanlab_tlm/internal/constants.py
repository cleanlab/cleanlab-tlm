from cleanlab_tlm.internal.types import Task, TLMQualityPreset

# TLM constants
# prepend constants with _ so that they don't show up in help.cleanlab.ai docs
_VALID_TLM_QUALITY_PRESETS: list[str] = ["best", "high", "medium", "low", "base"]
_VALID_TLM_QUALITY_PRESETS_RAG: list[str] = ["medium", "low", "base"]
_DEFAULT_TLM_QUALITY_PRESET: TLMQualityPreset = "medium"
_DEFAULT_TLM_MAX_TOKENS: int = 512
_VALID_TLM_MODELS: list[str] = [
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4.5-preview",
    "gpt-4o",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "o1-preview",
    "o1",
    "o1-mini",
    "o3",
    "o3-mini",
    "o4-mini",
    "claude-3-haiku",
    "claude-3.5-haiku",
    "claude-3-sonnet",
    "claude-3.5-sonnet",
    "claude-3.5-sonnet-v2",
    "claude-3.7-sonnet",
    "nova-micro",
    "nova-lite",
    "nova-pro",
]
_TLM_DEFAULT_MODEL: str = "gpt-4.1-mini"
_TLM_DEFAULT_CONTEXT_LIMIT: int = 70000
_VALID_TLM_TASKS: set[str] = {task.value for task in Task}
TLM_TASK_SUPPORTING_CONSTRAIN_OUTPUTS: set[Task] = {
    Task.DEFAULT,
    Task.CLASSIFICATION,
}
_TLM_MAX_RETRIES: int = 3  # TODO: finalize this number
_TLM_MAX_TOKEN_RANGE: dict[str, tuple[int, int]] = {  # model: (min, max)
    "default": (64, 4096),
    "claude-3-haiku": (64, 512),
    "claude-3.5-haiku": (64, 512),
    "claude-3-sonnet": (64, 512),
    "claude-3.5-sonnet": (64, 512),
    "nova-micro": (64, 512),
}
_TLM_CONSTRAIN_OUTPUTS_KEY: str = "constrain_outputs"
TLM_NUM_CANDIDATE_RESPONSES_RANGE: tuple[int, int] = (1, 20)  # (min, max)
TLM_NUM_CONSISTENCY_SAMPLES_RANGE: tuple[int, int] = (0, 20)  # (min, max)
TLM_SIMILARITY_MEASURES: set[str] = {
    "semantic",
    "string",
    "embedding",
    "embedding_large",
    "code",
    "discrepancy",
}
TLM_REASONING_EFFORT_VALUES: set[str] = {"none", "low", "medium", "high"}
TLM_VALID_LOG_OPTIONS: set[str] = {"perplexity", "explanation"}
TLM_VALID_GET_TRUSTWORTHINESS_SCORE_KWARGS: set[str] = {
    "perplexity",
    _TLM_CONSTRAIN_OUTPUTS_KEY,
}
TLM_VALID_PROMPT_KWARGS: set[str] = {_TLM_CONSTRAIN_OUTPUTS_KEY}
TLM_MODELS_NOT_SUPPORTING_EXPLANATION: set[str] = {
    "o1-mini",
    "o1-preview",
    "o1",
    "o3-mini",
    "o3",
    "o4-mini",
}
VALID_RESPONSE_OPTIONS: set[str] = {"max_tokens"}
INVALID_SCORE_OPTIONS: set[str] = {"num_candidate_responses"}

# API request and response field constants
_TLM_RESPONSE_KEY: str = "response"
_TLM_TRUSTWORTHINESS_KEY: str = "trustworthiness"
_TLM_QUALITY_KEY: str = "quality"
_TLM_OPTIONS_KEY: str = "options"
_TLM_USER_ID_KEY: str = "user_id"
_TLM_CLIENT_ID_KEY: str = "client_id"
_TLM_PROMPT_KEY: str = "prompt"
_TLM_QUERY_KEY: str = "query"
_TLM_CONTEXT_KEY: str = "context"
_TLM_EVALS_KEY: str = "evals"
_TLM_DEBERTA_SUCCESS_KEY: str = "deberta_success"
_TLM_TASK_KEY: str = "task"

# Evaluation-related constants
_TLM_EVAL_NAME_KEY: str = "name"
_TLM_EVAL_CRITERIA_KEY: str = "criteria"
_TLM_EVAL_QUERY_IDENTIFIER_KEY: str = "query_identifier"
_TLM_EVAL_CONTEXT_IDENTIFIER_KEY: str = "context_identifier"
_TLM_EVAL_RESPONSE_IDENTIFIER_KEY: str = "response_identifier"

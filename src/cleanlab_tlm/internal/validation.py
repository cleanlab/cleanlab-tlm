import os
from collections.abc import Sequence
from typing import Any, Optional, Union

from cleanlab_tlm.errors import ValidationError
from cleanlab_tlm.internal.constants import (
    _TLM_CONSTRAIN_OUTPUTS_KEY,
    _TLM_DEFAULT_MODEL,
    _TLM_MAX_TOKEN_RANGE,
    _VALID_TLM_MODELS,
    INVALID_SCORE_OPTIONS,
    TLM_MODELS_NOT_SUPPORTING_EXPLANATION,
    TLM_NUM_CANDIDATE_RESPONSES_RANGE,
    TLM_NUM_CONSISTENCY_SAMPLES_RANGE,
    TLM_REASONING_EFFORT_VALUES,
    TLM_SIMILARITY_MEASURES,
    TLM_TASK_SUPPORTING_CONSTRAIN_OUTPUTS,
    TLM_VALID_GET_TRUSTWORTHINESS_SCORE_KWARGS,
    TLM_VALID_LOG_OPTIONS,
    TLM_VALID_PROMPT_KWARGS,
    VALID_RESPONSE_OPTIONS,
)
from cleanlab_tlm.internal.types import Task

SKIP_VALIDATE_TLM_OPTIONS: bool = os.environ.get("CLEANLAB_TLM_SKIP_VALIDATE_TLM_OPTIONS", "false").lower() == "true"


def validate_tlm_prompt(prompt: Union[str, Sequence[str]]) -> None:
    if isinstance(prompt, str):
        return

    if isinstance(prompt, Sequence) and any(not isinstance(p, str) for p in prompt):
        raise ValidationError(
            "Some items in prompt are of invalid types, all items in the prompt list must be of type str."
        )


def validate_tlm_try_prompt(prompt: Sequence[str]) -> None:
    if isinstance(prompt, str):
        raise ValidationError("Invalid type str, prompt must be a list/iterable of strings.")

    if isinstance(prompt, Sequence) and any(not isinstance(p, str) for p in prompt):
        raise ValidationError(
            "Some items in prompt are of invalid types, all items in the prompt list must be of type str."
        )


def validate_tlm_prompt_response(prompt: Union[str, Sequence[str]], response: Union[str, Sequence[str]]) -> None:
    if isinstance(prompt, str):
        if not isinstance(response, str):
            raise ValidationError(
                "response type must match prompt type. "
                f"prompt was provided as str but response is of type {type(response)}"
            )

    elif isinstance(prompt, Sequence):
        # str is considered a Sequence, we want to explicitly check that response is not a string
        if not isinstance(response, Sequence) or isinstance(response, str):
            raise ValidationError(
                "response type must match prompt type. "
                f"prompt was provided as type {type(prompt)} but response is of type {type(response)}"
            )

        if len(prompt) != len(response):
            raise ValidationError("Length of the prompt and response lists must match.")

        if any(not isinstance(p, str) for p in prompt):
            raise ValidationError(
                "Some items in prompt are of invalid types, all items in the prompt list must be of type str."
            )
        if any(not isinstance(r, str) for r in response):
            raise ValidationError(
                "Some items in response are of invalid types, all items in the response list must be of type str."
            )


def validate_try_tlm_prompt_response(prompt: Sequence[str], response: Sequence[str]) -> None:
    if isinstance(prompt, str):
        raise ValidationError("Invalid type str, prompt must be a list/iterable of strings.")

    if isinstance(prompt, Sequence):
        if len(prompt) != len(response):
            raise ValidationError("Length of the prompt and response lists must match.")

        if any(not isinstance(p, str) for p in prompt):
            raise ValidationError(
                "Some items in prompt are of invalid types, all items in the prompt list must be of type str."
            )
        if any(not isinstance(r, str) for r in response):
            raise ValidationError(
                "Some items in response are of invalid types, all items in the response list must be of type str."
            )


def validate_tlm_options(options: Any) -> None:
    from cleanlab_tlm.tlm import TLMOptions

    if SKIP_VALIDATE_TLM_OPTIONS:
        return

    if not isinstance(options, dict):
        raise ValidationError(
            "options must be a TLMOptions object.\n"
            "See: https://help.cleanlab.ai/reference/python/tlm/#class-tlmoptions"
        )

    invalid_keys = set(options.keys()) - set(TLMOptions.__annotations__.keys())
    if invalid_keys:
        raise ValidationError(
            f"Invalid keys in options dictionary: {invalid_keys}.\n"
            "See https://help.cleanlab.ai/reference/python/tlm/#class-tlmoptions for valid options"
        )

    for option, val in options.items():
        if option == "max_tokens":
            if not isinstance(val, int):
                raise ValidationError(f"Invalid type {type(val)}, max_tokens must be an integer")

            model = options.get("model", _TLM_DEFAULT_MODEL)
            max_tokens_range = _TLM_MAX_TOKEN_RANGE.get(model, _TLM_MAX_TOKEN_RANGE["default"])
            if val < max_tokens_range[0] or val > max_tokens_range[1]:
                raise ValidationError(
                    f"Invalid value {val}, max_tokens for {model} must be in the range {max_tokens_range}"
                )

        elif option == "model":
            if val not in _VALID_TLM_MODELS:
                raise ValidationError(f"{val} is not a supported model, valid models include: {_VALID_TLM_MODELS}")

        elif option == "num_candidate_responses":
            if not isinstance(val, int):
                raise ValidationError(f"Invalid type {type(val)}, num_candidate_responses must be an integer")

            if val < TLM_NUM_CANDIDATE_RESPONSES_RANGE[0] or val > TLM_NUM_CANDIDATE_RESPONSES_RANGE[1]:
                raise ValidationError(
                    f"Invalid value {val}, num_candidate_responses must be in the range {TLM_NUM_CANDIDATE_RESPONSES_RANGE}"
                )

        elif option == "num_consistency_samples":
            if not isinstance(val, int):
                raise ValidationError(f"Invalid type {type(val)}, num_consistency_samples must be an integer")

            if val < TLM_NUM_CONSISTENCY_SAMPLES_RANGE[0] or val > TLM_NUM_CONSISTENCY_SAMPLES_RANGE[1]:
                raise ValidationError(
                    f"Invalid value {val}, num_consistency_samples must be in the range {TLM_NUM_CONSISTENCY_SAMPLES_RANGE}"
                )

        elif option == "use_self_reflection":
            if not isinstance(val, bool):
                raise ValidationError(f"Invalid type {type(val)}, use_self_reflection must be a boolean")

        elif option == "similarity_measure":
            if val not in TLM_SIMILARITY_MEASURES:
                raise ValidationError(
                    f"Invalid value for similarity_measure: {val}, valid measures include: {TLM_SIMILARITY_MEASURES}"
                )

        elif option == "reasoning_effort":
            if val not in TLM_REASONING_EFFORT_VALUES:
                raise ValidationError(
                    f"Invalid value {val}, reasoning_effort must be one of {TLM_REASONING_EFFORT_VALUES}"
                )

        elif option == "log":
            if not isinstance(val, list):
                raise ValidationError(f"Invalid type {type(val)}, log must be a list of strings.")

            invalid_log_options = set(val) - TLM_VALID_LOG_OPTIONS

            model = options.get("model", _TLM_DEFAULT_MODEL)
            if "explanation" in val and model in TLM_MODELS_NOT_SUPPORTING_EXPLANATION:
                raise ValidationError(f"Explanation is not supported for this model: {model}. ")

            if invalid_log_options:
                raise ValidationError(
                    f"Invalid options for log: {invalid_log_options}. Valid options include: {TLM_VALID_LOG_OPTIONS}"
                )


def process_and_validate_kwargs_constrain_outputs(
    prompt: Union[str, Sequence[str]],
    task: Task,
    kwargs_dict: dict[str, Any],
    response: Optional[Union[str, Sequence[str]]] = None,
) -> None:
    constrain_outputs = kwargs_dict.get(_TLM_CONSTRAIN_OUTPUTS_KEY)
    if constrain_outputs is None:
        if task == Task.CLASSIFICATION:
            raise ValidationError("constrain_outputs must be provided for classification tasks")

        return

    if task not in TLM_TASK_SUPPORTING_CONSTRAIN_OUTPUTS:
        raise ValidationError("constrain_outputs is only supported for classification tasks")

    if isinstance(prompt, str):
        if not isinstance(constrain_outputs, list) or not all(isinstance(s, str) for s in constrain_outputs):
            raise ValidationError("constrain_outputs must be a list of strings")
    elif isinstance(prompt, Sequence):
        if not isinstance(constrain_outputs, list):
            raise ValidationError("constrain_outputs must be a list")

        # If it's a list of strings, repeat the list for each prompt
        if all(isinstance(co, str) for co in constrain_outputs):
            constrain_outputs = [constrain_outputs] * len(prompt)
        # Check if it's a list of lists of strings
        elif all(isinstance(co, list) and all(isinstance(s, str) for s in co) for co in constrain_outputs):
            pass
        else:
            raise ValidationError("constrain_outputs must be a list of strings or a list of lists of strings")

        if len(constrain_outputs) != len(prompt):
            raise ValidationError("constrain_outputs must have same length as prompt")

    # Check each response is one of its allowed constraint outputs
    if response is not None:
        if isinstance(response, str):
            if response not in constrain_outputs:
                raise ValidationError(
                    f"Response '{response}' must be one of the constraint outputs: {constrain_outputs}"
                )
        else:
            for i, (resp, constraints) in enumerate(zip(response, constrain_outputs)):
                if constraints is not None and resp not in constraints:
                    raise ValidationError(
                        f"Response '{resp}' at index {i} must be one of the constraint outputs: {constraints}"
                    )

    kwargs_dict[_TLM_CONSTRAIN_OUTPUTS_KEY] = constrain_outputs


def tlm_prompt_process_and_validate_kwargs(
    prompt: Union[str, Sequence[str]],
    task: Task,
    kwargs_dict: dict[str, Any],
) -> None:
    if not SKIP_VALIDATE_TLM_OPTIONS:
        supported_kwargs = TLM_VALID_PROMPT_KWARGS
        unsupported_kwargs = set(kwargs_dict.keys()) - supported_kwargs
        if unsupported_kwargs:
            raise ValidationError(
                f"Unsupported keyword arguments: {unsupported_kwargs}. "
                f"Supported keyword arguments are: {supported_kwargs}"
            )

    process_and_validate_kwargs_constrain_outputs(prompt=prompt, task=task, kwargs_dict=kwargs_dict)


def tlm_score_process_response_and_kwargs(
    prompt: Union[str, Sequence[str]],
    response: Union[str, Sequence[str]],
    task: Task,
    kwargs_dict: dict[str, Any],
) -> Union[dict[str, Any], list[dict[str, Any]]]:
    process_and_validate_kwargs_constrain_outputs(prompt=prompt, task=task, kwargs_dict=kwargs_dict, response=response)

    if not SKIP_VALIDATE_TLM_OPTIONS:
        invalid_kwargs = set(kwargs_dict.keys()) - TLM_VALID_GET_TRUSTWORTHINESS_SCORE_KWARGS
        if invalid_kwargs:
            raise ValidationError(
                f"Invalid kwargs provided: {invalid_kwargs}. Valid kwargs include: {TLM_VALID_LOG_OPTIONS}"
            )
        # checking validity/format of each input kwarg, each one might require a different format
        for key, val in kwargs_dict.items():
            if key == "perplexity":
                if isinstance(response, str):
                    if not (isinstance(val, (float, int))):
                        raise ValidationError(
                            f"Invalid type {type(val)}, perplexity should be a float when response is a str."
                        )
                    if val is not None and not 0 <= val <= 1:
                        raise ValidationError("Perplexity values must be between 0 and 1")
                elif isinstance(response, Sequence):
                    if not isinstance(val, Sequence):
                        raise ValidationError(
                            f"Invalid type {type(val)}, perplexity should be a sequence when response is a sequence"
                        )
                    if len(response) != len(val):
                        raise ValidationError("Length of the response and perplexity lists must match.")
                    for v in val:
                        if not (isinstance(v, (float, int))):
                            raise ValidationError(f"Invalid type {type(v)}, perplexity values must be a float")
                        if v is not None and not 0 <= v <= 1:
                            raise ValidationError("Perplexity values must be between 0 and 1")
                else:
                    raise ValidationError(f"Invalid type {type(response)}, response must be either a sequence or a str")
    # format responses and kwargs into the appropriate formats
    combined_response = {"response": response, **kwargs_dict}
    if isinstance(response, str):
        return combined_response
    # else, there are multiple responses
    # transpose the dict of lists -> list of dicts, same length as prompt/response sequence
    combined_response_keys = combined_response.keys()
    combined_response_values_transposed = zip(*combined_response.values())
    return [dict(zip(combined_response_keys, values)) for values in combined_response_values_transposed]


def validate_tlm_lite_score_options(score_options: Any) -> None:
    invalid_score_keys = set(score_options.keys()).intersection(INVALID_SCORE_OPTIONS)
    if invalid_score_keys:
        raise ValidationError(
            f"Please remove these invalid keys from the `options` dictionary provided for TLMLite: {invalid_score_keys}.\n"
        )


def get_tlm_lite_response_options(score_options: Any, response_model: str) -> dict[str, Any]:
    response_options = {"model": response_model, "log": ["perplexity"]}
    if score_options is not None:
        for option_key in VALID_RESPONSE_OPTIONS:
            if option_key in score_options:
                response_options[option_key] = score_options[option_key]

    return response_options

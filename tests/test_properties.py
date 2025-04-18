import asyncio
from typing import Any, Union

import pytest

from cleanlab_tlm.internal.constants import _VALID_TLM_QUALITY_PRESETS
from cleanlab_tlm.internal.types import Task
from cleanlab_tlm.tlm import TLM
from tests.conftest import make_text_unique
from tests.constants import (
    MODELS_WITH_NO_PERPLEXITY_SCORE,
    TEST_CONSTRAIN_OUTPUTS,
    TEST_CONSTRAIN_OUTPUTS_BINARY,
    TEST_PROMPT,
    TEST_PROMPT_BATCH,
    TEST_RESPONSE,
    TEST_RESPONSE_BATCH,
    VALID_TLM_MODELS,
)
from tests.test_get_trustworthiness_score import (
    is_trustworthiness_score_json_format,
)
from tests.test_prompt import is_tlm_response

test_prompt_single = make_text_unique(TEST_PROMPT)
test_prompt_batch = [make_text_unique(prompt) for prompt in TEST_PROMPT_BATCH]


def _test_log(response: dict[str, Any], options: dict[str, Any]) -> None:
    """Tests the log dictionary in the response based on the options dictionary."""
    if "log" in options:
        print("Testing log:", options["log"], end="")
        if "log" in response:
            print(" response log:", response["log"], end="")
        else:
            print("... FAILED. NO LOG IN RESPONSE.")
        assert "log" in response
        assert isinstance(response["log"], dict)
        if "perplexity" in options["log"]:
            assert isinstance(response["log"]["perplexity"], float) or response["log"]["perplexity"] is None
        if "explanation" in options["log"]:
            assert isinstance(response["log"]["explanation"], str)
        print("... PASSED.")


def _test_log_batch(responses: list[dict[str, Any]], options: dict[str, Any]) -> None:
    """Tests the log dictionary in the batch response based on the options dictionary."""
    for response in responses:
        if response is not None:
            _test_log(response, options)


def _is_valid_prompt_response(
    response: dict[str, Any],
    options: dict[str, Any],
    allow_none_response: bool = False,
    allow_null_trustworthiness_score: bool = False,
) -> bool:
    """Returns true if prompt response is valid based on properties for prompt() functionality."""
    _test_log(response, options)
    if {"use_self_reflection", "num_consistency_samples"}.issubset(options) and (
        options["num_consistency_samples"] == 0 and not options["use_self_reflection"]
    ):
        print("Options dictinary called with strange parameters. Allowing none in response.")
        return is_tlm_response(
            response,
            allow_none_response=allow_none_response,
            allow_null_trustworthiness_score=True,
        )
    return is_tlm_response(
        response,
        allow_none_response=allow_none_response,
        allow_null_trustworthiness_score=allow_null_trustworthiness_score,
    )


def _is_valid_get_trustworthiness_score_response(
    response: dict[str, Any],
    options: dict[str, Any],
    quality_preset: str,
    allow_null_trustworthiness_score: bool = False,
) -> bool:
    """Returns true if trustworthiness score is valid based on properties for get_trustworthiness_score() functionality."""
    assert isinstance(response, dict)

    quality_preset_keys = {"use_self_reflection"}
    consistency_sample_keys = {"num_consistency_samples", "use_self_reflection"}

    if (
        (quality_preset_keys.issubset(options)) and not options["use_self_reflection"] and quality_preset == "base"
    ) or (
        (consistency_sample_keys.issubset(options))
        and not options["use_self_reflection"]
        and options["num_consistency_samples"] == 0
    ):
        print("Options dictinary called with strange parameters. Allowing none in response.")
        return is_trustworthiness_score_json_format(response, allow_null_trustworthiness_score=True)
    return is_trustworthiness_score_json_format(
        response,
        allow_null_trustworthiness_score=allow_null_trustworthiness_score,
    )


def _test_prompt_response(
    response: dict[str, Any],
    options: dict[str, Any],
    allow_none_response: bool = False,
    allow_null_trustworthiness_score: bool = False,
) -> None:
    """Property tests the responses of a prompt based on the options dictionary and returned responses."""
    assert _is_valid_prompt_response(
        response=response,
        options=options,
        allow_none_response=allow_none_response,
        allow_null_trustworthiness_score=allow_null_trustworthiness_score,
    )


def _test_batch_prompt_response(
    responses: list[dict[str, Any]],
    options: dict[str, Any],
    allow_none_response: bool = False,
    allow_null_trustworthiness_score: bool = False,
) -> None:
    """Property tests the responses of a batch prompt based on the options dictionary and returned responses."""
    assert responses is not None
    assert isinstance(responses, list)
    _test_log_batch(responses, options)

    checked_responses = [
        _is_valid_prompt_response(
            response,
            options,
            allow_none_response=allow_none_response,
            allow_null_trustworthiness_score=allow_null_trustworthiness_score,
        )
        for response in responses
    ]
    print("Checked respones:", checked_responses)
    assert all(checked_responses)


def _test_get_trustworthiness_score_response(
    response: dict[str, Any],
    options: dict[str, Any],
    quality_preset: str,
    allow_null_trustworthiness_score: bool = False,
) -> None:
    """Property tests the responses of a get_trustworthiness_score based on the options dictionary and returned responses."""
    assert _is_valid_get_trustworthiness_score_response(
        response=response,
        options=options,
        quality_preset=quality_preset,
        allow_null_trustworthiness_score=allow_null_trustworthiness_score,
    )


def _test_batch_get_trustworthiness_score_response(
    responses: list[dict[str, Any]],
    options: dict[str, Any],
    quality_preset: str,
    allow_null_trustworthiness_score: bool = False,
) -> None:
    """Property tests the responses of a batch get_trustworthiness_score based on the options dictionary and returned responses."""
    assert responses is not None
    assert isinstance(responses, list)
    _test_log_batch(responses, options)

    checked_responses = [
        _is_valid_get_trustworthiness_score_response(
            response,
            options,
            quality_preset=quality_preset,
            allow_null_trustworthiness_score=allow_null_trustworthiness_score,
        )
        for response in responses
    ]
    print("Checked respones:", checked_responses)
    assert all(checked_responses)


@pytest.mark.asyncio(scope="function")
async def _run_prompt_async(tlm: TLM, prompt: Union[list[str], str], **kwargs: Any) -> Any:
    """Runs tlm.prompt() asynchronously."""
    return await tlm.prompt_async(prompt, **kwargs)


@pytest.mark.asyncio(scope="function")
async def _run_get_trustworthiness_score_async(
    tlm: TLM,
    prompt: Union[list[str], str],
    response: Union[list[str], str],
    **kwargs: Any,
) -> Any:
    """Runs tlm.get_trustworthiness_score asynchronously."""
    return await tlm.get_trustworthiness_score_async(prompt, response, **kwargs)


@pytest.mark.parametrize("model", VALID_TLM_MODELS)
@pytest.mark.parametrize("quality_preset", _VALID_TLM_QUALITY_PRESETS)
def test_prompt(tlm_dict: dict[str, Any], model: str, quality_preset: str) -> None:
    """Tests running a prompt in the TLM for all quality_presets, model types and single/batch prompt."""
    print("Testing with prompt:", test_prompt_single)
    print("Testing with batch prompt:", test_prompt_batch)
    # get TLMs and options dictionary based on parameters
    tlm = tlm_dict[quality_preset][model]["tlm"]
    tlm_no_options = tlm_dict[quality_preset][model]["tlm_no_options"]
    options = tlm_dict[quality_preset][model]["options"]
    allow_null_trustworthiness_score = quality_preset == "base" and model in MODELS_WITH_NO_PERPLEXITY_SCORE
    print("TLM with no options called on single query run.")
    print("TLM Options for run:", options)

    # test prompt with single prompt
    tlm_no_options_kwargs = {}
    if tlm_no_options._task == Task.CLASSIFICATION:
        tlm_no_options_kwargs["constrain_outputs"] = TEST_CONSTRAIN_OUTPUTS_BINARY
    response = tlm_no_options.prompt(test_prompt_single, **tlm_no_options_kwargs)
    print("TLM Single Response:", response)
    _test_prompt_response(
        response,
        {},
        allow_null_trustworthiness_score=allow_null_trustworthiness_score,
    )

    # test prompt with batch prompt
    tlm_kwargs = {}
    if tlm._task == Task.CLASSIFICATION:
        tlm_kwargs["constrain_outputs"] = TEST_CONSTRAIN_OUTPUTS
    responses = tlm.prompt(test_prompt_batch, **tlm_kwargs)
    print("TLM Batch Responses:", responses)
    _test_batch_prompt_response(
        responses,
        options,
        allow_none_response=True,
        allow_null_trustworthiness_score=allow_null_trustworthiness_score,
    )


@pytest.mark.parametrize("model", VALID_TLM_MODELS)
@pytest.mark.parametrize("quality_preset", _VALID_TLM_QUALITY_PRESETS)
def test_prompt_async(tlm_dict: dict[str, Any], model: str, quality_preset: str) -> None:
    """Tests running a prompt_async in the TLM for all quality_presets, model types and single/batch prompt."""
    print("Testing with prompt:", test_prompt_single)
    print("Testing with batch prompt:", test_prompt_batch)
    # get TLMs and options dictionary based on parameters
    tlm = tlm_dict[quality_preset][model]["tlm"]
    tlm_no_options = tlm_dict[quality_preset][model]["tlm_no_options"]
    options = tlm_dict[quality_preset][model]["options"]
    allow_null_trustworthiness_score = quality_preset == "base" and model in MODELS_WITH_NO_PERPLEXITY_SCORE
    print("TLM with no options called on single query run.")
    print("TLM Options for run:", options)

    # test prompt with single prompt
    tlm_no_options_kwargs = {}
    if tlm_no_options._task == Task.CLASSIFICATION:
        tlm_no_options_kwargs["constrain_outputs"] = TEST_CONSTRAIN_OUTPUTS_BINARY
    response = asyncio.run(_run_prompt_async(tlm_no_options, test_prompt_single, **tlm_no_options_kwargs))
    print("TLM Single Response:", response)
    _test_prompt_response(response, {}, allow_null_trustworthiness_score=allow_null_trustworthiness_score)

    # test prompt with batch prompt
    tlm_kwargs = {}
    if tlm._task == Task.CLASSIFICATION:
        tlm_kwargs["constrain_outputs"] = TEST_CONSTRAIN_OUTPUTS
    responses = asyncio.run(_run_prompt_async(tlm, test_prompt_batch, **tlm_kwargs))
    print("TLM Batch Responses:", responses)
    _test_batch_prompt_response(
        responses,
        options,
        allow_null_trustworthiness_score=allow_null_trustworthiness_score,
    )


@pytest.mark.parametrize("model", VALID_TLM_MODELS)
@pytest.mark.parametrize("quality_preset", _VALID_TLM_QUALITY_PRESETS)
def test_get_trustworthiness_score(tlm_dict: dict[str, Any], model: str, quality_preset: str) -> None:
    """Tests running get_trustworthiness_score in the TLM for all quality_presets, model types and single/batch prompt."""
    print("Testing with prompt/response:", test_prompt_single, TEST_RESPONSE)
    print("Testing with batch prompt/response:", test_prompt_batch, TEST_RESPONSE_BATCH)
    # get TLMs and options dictionary based on parameters
    tlm = tlm_dict[quality_preset][model]["tlm"]
    tlm_no_options = tlm_dict[quality_preset][model]["tlm_no_options"]
    options = tlm_dict[quality_preset][model]["options"]
    print("TLM with no options called on batch query run.")
    print("TLM Options for run:", options)

    # test prompt with single prompt
    tlm_kwargs = {}
    if tlm._task == Task.CLASSIFICATION:
        tlm_kwargs["constrain_outputs"] = TEST_CONSTRAIN_OUTPUTS_BINARY
    response = tlm.get_trustworthiness_score(test_prompt_single, TEST_RESPONSE, **tlm_kwargs)
    print("TLM Single Response:", response)
    _test_get_trustworthiness_score_response(response, options, quality_preset)

    # test prompt with batch prompt
    tlm_no_options_kwargs = {}
    if tlm_no_options._task == Task.CLASSIFICATION:
        tlm_no_options_kwargs["constrain_outputs"] = TEST_CONSTRAIN_OUTPUTS
    responses = tlm_no_options.get_trustworthiness_score(
        test_prompt_batch, TEST_RESPONSE_BATCH, **tlm_no_options_kwargs
    )
    print("TLM Batch Responses:", responses)
    _test_batch_get_trustworthiness_score_response(responses, {}, quality_preset)


@pytest.mark.parametrize("model", VALID_TLM_MODELS)
@pytest.mark.parametrize("quality_preset", _VALID_TLM_QUALITY_PRESETS)
def test_get_trustworthiness_score_async(tlm_dict: dict[str, Any], model: str, quality_preset: str) -> None:
    """Tests running get_trustworthiness_score_async in the TLM for all quality_presets, model types and single/batch prompt."""
    print("Testing with prompt/response:", test_prompt_single, TEST_RESPONSE)
    print("Testing with batch prompt/response:", test_prompt_batch, TEST_RESPONSE_BATCH)
    # get TLMs and options dictionary based on parameters
    tlm = tlm_dict[quality_preset][model]["tlm"]
    tlm_no_options = tlm_dict[quality_preset][model]["tlm_no_options"]
    options = tlm_dict[quality_preset][model]["options"]
    print("TLM with no options called on single query run.")
    print("TLM Options for run:", options)

    # test prompt with single prompt
    tlm_no_options_kwargs = {}
    if tlm_no_options._task == Task.CLASSIFICATION:
        tlm_no_options_kwargs["constrain_outputs"] = TEST_CONSTRAIN_OUTPUTS_BINARY
    response = asyncio.run(
        _run_get_trustworthiness_score_async(tlm_no_options, test_prompt_single, TEST_RESPONSE, **tlm_no_options_kwargs)
    )
    print("TLM Single Response:", response)
    _test_get_trustworthiness_score_response(response, {}, quality_preset)

    # test prompt with batch prompt
    tlm_kwargs = {}
    if tlm._task == Task.CLASSIFICATION:
        tlm_kwargs["constrain_outputs"] = TEST_CONSTRAIN_OUTPUTS
    responses = asyncio.run(
        _run_get_trustworthiness_score_async(
            tlm,
            test_prompt_batch,
            TEST_RESPONSE_BATCH,
            **tlm_kwargs,
        )
    )
    print("TLM Batch Responses:", responses)
    _test_batch_get_trustworthiness_score_response(responses, options, quality_preset)

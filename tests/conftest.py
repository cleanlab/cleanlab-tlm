import os
import random
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pytest
from dotenv import load_dotenv

from cleanlab_tlm.errors import MissingApiKeyError
from cleanlab_tlm.internal.concurrency import TlmRateHandler
from cleanlab_tlm.internal.constants import (
    _TLM_DEFAULT_MODEL,
    _TLM_MAX_TOKEN_RANGE,
    _VALID_TLM_MODELS,
    _VALID_TLM_TASKS,
    TLM_MODELS_NOT_SUPPORTING_EXPLANATION,
    TLM_REASONING_EFFORT_VALUES,
    TLM_SIMILARITY_MEASURES,
)
from cleanlab_tlm.internal.types import TLMQualityPreset
from cleanlab_tlm.tlm import TLM, TLMOptions

load_dotenv()


@pytest.fixture(scope="module")
def tlm_api_key() -> str:
    api_key = os.environ.get("CLEANLAB_TLM_API_KEY")
    if api_key is None:
        raise MissingApiKeyError
    return api_key


@pytest.fixture(scope="module")
def tlm(tlm_api_key: str) -> TLM:
    """Creates a TLM with default settings."""
    try:
        # uses environment API key
        return TLM(api_key=tlm_api_key)
    except Exception as e:
        environment = os.environ.get("CLEANLAB_API_BASE_URL")
        pytest.skip(f"Failed to create TLM: {e}. Check your API key and environment: ({environment}).")


@pytest.fixture(scope="module")
def tlm_dict(tlm_api_key: str) -> dict[str, Any]:
    """Creates a dictionary of initialized tlm objects for each quality preset and model to be reused throughout the test.
    Save randomly created options dictionary for each tlm object as well.

    Initializes two TLM objects for each quality preset and model:
    - One with randomly generated options
    - One with default presets (no options)

    Each function call is tested on both of these TLM objects to ensure that the function works with options and for the default preset
    and to give signal if the function is not working for a specific set of options or overall.
    """

    tlm_dict: dict[str, Any] = {}
    for quality_preset in TLMQualityPreset.__args__:  # type: ignore
        tlm_dict[quality_preset] = {}
        for model in _VALID_TLM_MODELS:
            tlm_dict[quality_preset][model] = {}
            task = random.choice(list(_VALID_TLM_TASKS))
            options = _get_options_dictionary(model)
            tlm_dict[quality_preset][model]["tlm"] = TLM(
                quality_preset=quality_preset,
                task=task,
                api_key=tlm_api_key,
                options=options,
            )
            tlm_dict[quality_preset][model]["tlm_no_options"] = TLM(
                quality_preset=quality_preset,
                task=task,
                api_key=tlm_api_key,
            )
            tlm_dict[quality_preset][model]["options"] = options
    return tlm_dict


@pytest.fixture
def tlm_rate_handler() -> TlmRateHandler:
    """Creates a TlmRateHandler with default settings."""
    return TlmRateHandler()


def _get_options_dictionary(model: Optional[str]) -> TLMOptions:
    """Returns randomly generated TLMOptions for the TLM."""
    add_max_tokens = np.random.choice([True, False])
    add_num_candidate_responses = np.random.choice([True, False])
    add_num_consistency_samples = np.random.choice([True, False])
    add_use_self_reflection = np.random.choice([True, False])
    add_similarity_measure = np.random.choice([True, False])
    add_reasoning_effort = np.random.choice([True, False])
    add_log_explanation = (
        np.random.choice([True, False]) if model not in TLM_MODELS_NOT_SUPPORTING_EXPLANATION else False
    )
    add_log_perplexity_score = np.random.choice([True, False])

    options: dict[str, Any] = {}

    if model is not None:
        options["model"] = model

    if add_max_tokens:
        max_tokens_limit = _TLM_MAX_TOKEN_RANGE.get(model or _TLM_DEFAULT_MODEL, _TLM_MAX_TOKEN_RANGE["default"])[1]
        options["max_tokens"] = int(np.random.randint(64, max_tokens_limit))
    if add_use_self_reflection:
        options["use_self_reflection"] = random.choice([True, False])
    if add_num_candidate_responses:
        options["num_candidate_responses"] = int(np.random.randint(1, 5))
    if add_num_consistency_samples:
        options["num_consistency_samples"] = int(np.random.randint(0, 10))
    if add_similarity_measure:
        options["similarity_measure"] = random.choice(list(TLM_SIMILARITY_MEASURES))
    if add_reasoning_effort:
        options["reasoning_effort"] = random.choice(list(TLM_REASONING_EFFORT_VALUES))

    if add_log_explanation or add_log_perplexity_score:
        options["log"] = [
            key
            for key, options_flag in {
                "explanation": add_log_explanation,
                "perplexity": add_log_perplexity_score,
            }.items()
            if options_flag
        ]
    return TLMOptions(**options)  # type: ignore


def make_text_unique(text: str) -> str:
    """Makes a text unique by prepending the curent datatime to it."""
    return str(datetime.now().strftime("%Y%m%d%H%M%S")) + " " + text

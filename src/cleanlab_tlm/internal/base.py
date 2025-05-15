"""
Base class for TLM-based classes with shared initialization logic.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import TYPE_CHECKING, Optional, Union

from cleanlab_tlm.errors import MissingApiKeyError, ValidationError
from cleanlab_tlm.internal.concurrency import TlmRateHandler
from cleanlab_tlm.internal.constants import _TLM_DEFAULT_MODEL
from cleanlab_tlm.internal.validation import validate_tlm_options

if TYPE_CHECKING:
    from cleanlab_tlm.tlm import TLMOptions


def is_notebook() -> bool:
    """Returns True if running in a notebook, False otherwise."""
    try:
        get_ipython = sys.modules["IPython"].get_ipython
        return bool("IPKernelApp" in get_ipython().config)
    except Exception:
        return False


class BaseTLM:
    """Base class for TLM-based classes with shared initialization logic."""

    def __init__(
        self,
        quality_preset: str,
        valid_quality_presets: list[str],
        support_custom_eval_criteria: bool,
        *,
        api_key: Optional[str] = None,
        options: Optional[TLMOptions] = None,
        timeout: Optional[float] = None,
        verbose: Optional[bool] = None,
    ) -> None:
        """
        Initialize base TLM functionality.

        Args:
            quality_preset: Preset configuration to control the quality of trustworthiness scores.
            valid_quality_presets: List of valid quality presets for validation.
            support_custom_eval_criteria: Whether this class supports custom evaluation criteria.
            api_key: API key for accessing the TLM service.
            options: A typed dict of advanced configuration options.
            timeout: Timeout (in seconds) to apply to each TLM prompt.
            verbose: Whether to print outputs during execution.
        """
        self._api_key = api_key or os.environ.get("CLEANLAB_TLM_API_KEY")
        if self._api_key is None:
            raise MissingApiKeyError

        if quality_preset not in valid_quality_presets:
            raise ValidationError(f"Invalid quality preset {quality_preset} -- must be one of {valid_quality_presets}")

        self._return_log = False

        options_dict = options or {}
        validate_tlm_options(options_dict, support_custom_eval_criteria)
        if "log" in options_dict and len(options_dict["log"]) > 0:
            self._return_log = True

        if "custom_eval_criteria" in options_dict:
            self._return_log = True

        # explicitly specify the default model
        self._options = {"model": _TLM_DEFAULT_MODEL, **options_dict}

        self._quality_preset = quality_preset

        self._timeout: Optional[Union[int, float]] = None
        if timeout is not None:
            if not isinstance(timeout, (float, int)):
                raise ValidationError("timeout must be a integer or float value")
            if timeout <= 0:
                raise ValidationError("timeout must be a positive value")
            self._timeout = timeout
            self._options["max_timeout"] = self._timeout

        if verbose is not None and not isinstance(verbose, bool):
            raise ValidationError("verbose must be a boolean value")

        is_notebook_flag = is_notebook()

        self._verbose = verbose if verbose is not None else is_notebook_flag

        if is_notebook_flag:
            import nest_asyncio  # type: ignore

            nest_asyncio.apply()

        try:
            self._event_loop = asyncio.get_event_loop()
        except RuntimeError:
            self._event_loop = asyncio.new_event_loop()
        self._rate_handler = TlmRateHandler()

"""
Shared exception handling utilities for TLM modules.
"""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import Coroutine
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, cast

from cleanlab_tlm.errors import (
    APITimeoutError,
    RateLimitError,
    TlmBadRequestError,
    TlmServerError,
)
from cleanlab_tlm.internal.types import TLMResult

# Define type variables for the response types
ResponseT = TypeVar("ResponseT")
ResponseTypes = TypeVar("ResponseTypes")


def handle_tlm_exceptions(
    response_type: str,
) -> Callable[
    [Callable[..., Coroutine[Any, Any, ResponseT]]],
    Callable[..., Coroutine[Any, Any, ResponseT]],
]:
    """Decorator to handle exceptions for TLM API calls.
    
    This decorator can be used with any async function that returns a TLM response type.
    It catches various exceptions that might occur during API calls and handles them
    appropriately based on the capture_exceptions flag.
    
    Args:
        response_type (str): The type of response expected from the decorated function.
            This should be one of "TLMResponse", "TLMScore", "TrustworthyRAGResponse", 
            or "TrustworthyRAGScore".
            
    Returns:
        A decorator function that wraps an async function to handle exceptions.
    """

    def decorator(
        func: Callable[..., Coroutine[Any, Any, ResponseT]],
    ) -> Callable[..., Coroutine[Any, Any, ResponseT]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> ResponseT:
            capture_exceptions = kwargs.get("capture_exceptions", False)
            batch_index = kwargs.get("batch_index")
            try:
                return await func(*args, **kwargs)
            except asyncio.TimeoutError:
                return cast(
                    ResponseT,
                    _handle_exception(
                        APITimeoutError(
                            "Timeout while waiting for prediction. Please retry or consider increasing the timeout."
                        ),
                        capture_exceptions,
                        batch_index,
                        retryable=True,
                        response_type=response_type,
                    ),
                )
            except RateLimitError as e:
                return cast(
                    ResponseT,
                    _handle_exception(
                        e,
                        capture_exceptions,
                        batch_index,
                        retryable=True,
                        response_type=response_type,
                    ),
                )
            except TlmBadRequestError as e:
                return cast(
                    ResponseT,
                    _handle_exception(
                        e,
                        capture_exceptions,
                        batch_index,
                        retryable=e.retryable,
                        response_type=response_type,
                    ),
                )
            except TlmServerError as e:
                return cast(
                    ResponseT,
                    _handle_exception(
                        e,
                        capture_exceptions,
                        batch_index,
                        retryable=True,
                        response_type=response_type,
                    ),
                )
            except Exception as e:
                return cast(
                    ResponseT,
                    _handle_exception(
                        e,
                        capture_exceptions,
                        batch_index,
                        retryable=True,
                        response_type=response_type,
                    ),
                )

        return wrapper

    return decorator


def _handle_exception(
    e: Exception,
    capture_exceptions: bool,
    batch_index: Optional[int],
    retryable: bool,
    response_type: str,
) -> TLMResult:
    if capture_exceptions:
        retry_message = (
            "Worth retrying."
            if retryable
            else "Retrying will not help. Please address the issue described in the error message before attempting again."
        )
        error_message = str(e.message) if hasattr(e, "message") else str(e)
        warning_message = f"prompt[{batch_index}] failed. {retry_message} Error: {error_message}"
        warnings.warn(warning_message)

        error_log = {"error": {"message": error_message, "retryable": retryable}}

        if response_type == "TLMResponse":
            # Import here to avoid circular imports
            from cleanlab_tlm.tlm import TLMResponse
            return TLMResponse(
                response=None,
                trustworthiness_score=None,
                log=error_log,
            )
        if response_type == "TLMScore":
            # Import here to avoid circular imports
            from cleanlab_tlm.tlm import TLMScore
            return TLMScore(
                trustworthiness_score=None,
                log=error_log,
            )
        if response_type == "TrustworthyRAGResponse":
            # Import here to avoid circular imports
            from cleanlab_tlm.utils.tlm_rag import TrustworthyRAGResponse
            return TrustworthyRAGResponse(
                response=None,
                trustworthiness_score=None,
                log=error_log,
            )
        if response_type == "TrustworthyRAGScore":
            # Import here to avoid circular imports
            from cleanlab_tlm.utils.tlm_rag import TrustworthyRAGScore
            return TrustworthyRAGScore(
                trustworthiness_score=None,
                log=error_log,
            )

        raise ValueError(f"Unsupported response type: {response_type}")

    if len(e.args) > 0:
        additional_message = "Consider using `TLM.try_prompt()` or `TLM.try_get_trustworthiness_score()` to gracefully handle errors and preserve partial results. For large datasets, consider also running it on multiple smaller batches."
        new_args = (str(e.args[0]) + "\n" + additional_message,) + e.args[1:]
        raise type(e)(*new_args)

    raise e  # in the case where the error has no message/args

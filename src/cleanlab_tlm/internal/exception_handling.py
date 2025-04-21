"""
Shared exception handling utilities for TLM modules.
"""

from __future__ import annotations

import asyncio
import warnings
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union, cast

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from cleanlab_tlm.internal.types import TLMResult

from cleanlab_tlm.errors import (
    APITimeoutError,
    AuthError,
    RateLimitError,
    TlmBadRequestError,
    TlmServerError,
)

# Define type variables for the response types
ResponseT = TypeVar("ResponseT")
ResponseTypes = TypeVar("ResponseTypes")

RAG_GENERATE_RESPONSE_TYPE = "TrustworthyRAGResponse"
RAG_SCORE_RESPONSE_TYPE = "TrustworthyRAGScore"


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
            evals = getattr(args[0], "_evals", [])
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
                        evals=evals,
                    ),
                )
            except AuthError as e:
                return cast(
                    ResponseT,
                    _handle_exception(
                        e,
                        capture_exceptions,
                        batch_index,
                        retryable=False,
                        response_type=response_type,
                        evals=evals,
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
                        evals=evals,
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
                        evals=evals,
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
                        evals=evals,
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
                        evals=evals,
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
    evals: Optional[list[Any]] = None,
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

        # Helper function to create evaluation metrics dictionary
        def create_eval_metrics(
            include_response: bool = False,
        ) -> dict[str, Union[dict[str, Any], None]]:
            result: dict[str, Union[dict[str, Any], None]] = {
                "trustworthiness": {
                    "score": None,
                    "log": error_log,
                },
            }

            # Add response field if needed
            if include_response:
                result["response"] = None

            # Add all evaluation metrics from evals list
            if evals:
                for eval_obj in evals:
                    if hasattr(eval_obj, "name") and eval_obj.name != "trustworthiness":
                        result[eval_obj.name] = {"score": None}

            return result

        if response_type == "TLMResponse":
            # Return a dictionary matching TLMResponse TypedDict structure
            return {
                "response": None,
                "trustworthiness_score": None,
                "log": error_log,
            }
        if response_type == "TLMScore":
            # Return a dictionary matching TLMScore TypedDict structure
            return {
                "trustworthiness_score": None,
                "log": error_log,
            }

        if response_type == RAG_GENERATE_RESPONSE_TYPE:
            # Return a dictionary matching TrustworthyRAGResponse structure
            return create_eval_metrics(include_response=True)

        if response_type == RAG_SCORE_RESPONSE_TYPE:
            # Return a dictionary matching TrustworthyRAGScore structure
            return create_eval_metrics(include_response=False)

        raise ValueError(f"Unsupported response type: {response_type}")

    raise e  # in the case where the error has no message/args

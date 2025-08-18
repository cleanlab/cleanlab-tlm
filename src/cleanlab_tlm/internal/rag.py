from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from cleanlab_tlm.utils.chat import _TOOL_CALL_TAG_END, _TOOL_CALL_TAG_START

if TYPE_CHECKING:
    from collections.abc import Coroutine

# Define type variables for the response types
ResponseT = TypeVar("ResponseT")


def _is_tool_call_response(response_string: str) -> bool:
    """Check if response string represents a tool call."""
    stripped = response_string.strip()

    # If response doesn't contain tool call tags, it's not a tool call
    if _TOOL_CALL_TAG_START not in stripped or _TOOL_CALL_TAG_END not in stripped:
        return False

    # Find all tool call sections and remove them
    remaining_content = stripped
    while _TOOL_CALL_TAG_START in remaining_content and _TOOL_CALL_TAG_END in remaining_content:
        start_pos = remaining_content.find(_TOOL_CALL_TAG_START)
        end_pos = remaining_content.find(_TOOL_CALL_TAG_END, start_pos)

        # If we can't find a matching closing tag, break
        if end_pos == -1:
            break

        # Remove this tool call section (including the tags)
        end_pos += len(_TOOL_CALL_TAG_END)
        remaining_content = remaining_content[:start_pos] + remaining_content[end_pos:]

    # If there's any non-whitespace content left after removing all tool calls,
    # then this response contains regular text and is not a pure tool call response
    return not remaining_content.strip()


def _handle_tool_call_filtering(
    func: Callable[..., Coroutine[Any, Any, ResponseT]],
) -> Callable[..., Coroutine[Any, Any, ResponseT]]:
    """
    Decorator to handle tool call filtering for scoring methods.

    When tool call handling is enabled and a tool call is detected:
    - Filters out evals that have response_identifier (these would get None scores)
    - Calls the original method with filtered evals via a context wrapper
    - Adds None scores for the filtered evals
    - Returns the combined result

    This implementation avoids modifying the original instance state to prevent
    race conditions in concurrent async operations.
    """

    @wraps(func)
    async def wrapper(self: Any, **kwargs: Any) -> ResponseT:
        response = kwargs.get("response", {})
        response_text = response.get("response", "")
        is_tool_call = _is_tool_call_response(str(response_text))

        # If not a tool call, just call the original method
        if not is_tool_call:
            return await func(self, **kwargs)

        # It's a tool call - determine which evals to process vs. filter
        # Default behavior:
        #   - Evals with response_identifier are filtered out (score None)
        #   - Evals without response_identifier are still evaluated normally
        # Optional per-eval overrides via instance-level include/exclude name sets:
        #   - If name in exclude set, filter (score None)

        exclude_names = set(getattr(self, "_tool_call_eval_exclude_names", set()) or set())

        evals_to_process = []
        tool_call_filtered_evals = []

        for eval_obj in self._evals:
            # Start from default filtering decision
            is_filtered = eval_obj.response_identifier is not None and eval_obj.name in exclude_names

            if is_filtered:
                tool_call_filtered_evals.append(eval_obj)
            else:
                evals_to_process.append(eval_obj)

        # Create a context wrapper that temporarily provides filtered evals
        # without modifying the original instance
        class _EvalsContextWrapper:
            def __init__(self, original_instance: Any, filtered_evals: list[Any]):
                self._original = original_instance
                self._filtered_evals = filtered_evals

            def __getattr__(self, name: str) -> Any:
                if name == "_evals":
                    return self._filtered_evals
                return getattr(self._original, name)

            def __repr__(self) -> str:
                return repr(self._original)

            def __str__(self) -> str:
                return str(self._original)

        # Use the wrapper instance to call the original method
        wrapper_instance = _EvalsContextWrapper(self, evals_to_process)
        backend_response: ResponseT = await func(wrapper_instance, **kwargs)
        return _rebuild_response(backend_response, self._evals)

    return wrapper


def _rebuild_response(backend_response: ResponseT, evals: list[Any]) -> ResponseT:
    eval_names = [e.name for e in evals]
    ordered = {}

    for k, v in backend_response.items():  # type: ignore
        if k not in eval_names:
            ordered[k] = v

    for e in evals:
        name = e.name
        if name in backend_response:  # type: ignore
            ordered[name] = backend_response[name]  # type: ignore
        else:
            ordered[name] = {"score": None}  # filtered or missing

    return ordered  # type: ignore

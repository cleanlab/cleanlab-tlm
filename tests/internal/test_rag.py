from typing import Any
from unittest import mock

from cleanlab_tlm.utils.rag import TrustworthyRAG
from tests.test_tlm_rag import (
    test_context,
    test_prompt,
    test_query,
    test_response,
    trustworthy_rag,  # noqa: F401
    trustworthy_rag_api_key,  # noqa: F401
)


def test_decorator_skips_bulk_logic_for_non_tool_calls(trustworthy_rag: TrustworthyRAG) -> None:  # noqa: F811
    """Tests that the _handle_tool_call_filtering decorator skips the bulk of its logic for non-tool calls.

    Expected:
    - When _is_tool_call_response returns False, the decorator should skip eval filtering logic
    - The original _evals should not be modified during execution
    - No None scores should be added for tool call filtered evals
    """
    # Store original evals for comparison
    original_evals = trustworthy_rag._evals.copy()
    original_evals_id = id(trustworthy_rag._evals)

    # Mock to track if the bulk logic is executed
    with mock.patch("cleanlab_tlm.internal.rag._is_tool_call_response", return_value=False) as mock_is_tool_call:
        # Track if evals are temporarily modified (which shouldn't happen for non-tool calls)
        evals_modifications = []
        original_setattr = object.__setattr__

        def track_evals_setattr(self: Any, name: str, value: Any) -> Any:
            if name == "_evals" and hasattr(self, "_evals"):
                evals_modifications.append((name, value, id(value)))
            return original_setattr(self, name, value)

        with mock.patch.object(type(trustworthy_rag), "__setattr__", track_evals_setattr):
            response = trustworthy_rag.score(
                query=test_query,
                context=test_context,
                response=test_response,
                prompt=test_prompt,
            )

        # Verify _is_tool_call_response was called (decorator logic was entered)
        assert mock_is_tool_call.call_count > 0

        # Verify that evals were not temporarily modified (bulk logic was skipped)
        # The only modifications should be the initial assignment during init, not temporary changes
        evals_temp_modifications = [mod for mod in evals_modifications if mod[2] != original_evals_id]
        assert len(evals_temp_modifications) == 0, f"Evals were temporarily modified: {evals_temp_modifications}"

    # Verify evals are unchanged after the call
    assert trustworthy_rag._evals == original_evals
    assert id(trustworthy_rag._evals) == original_evals_id

    # Verify we got a normal response with actual scores (not None scores from tool call filtering)
    assert isinstance(response, dict)
    for eval_name, eval_data in response.items():
        if eval_name != "trustworthiness":  # trustworthiness might have None score if disabled
            # Non-tool calls should have actual scores, not None scores from tool call filtering
            assert eval_data["score"] is not None or eval_name == "trustworthiness"


def test_decorator_calls_api_with_full_evals_for_non_tool_calls(trustworthy_rag_api_key: str) -> None:  # noqa: F811
    """Decorator should pass full evals to API for non-tool-call responses.

    Expected:
    - When _is_tool_call_response returns False, the decorator should call the underlying API
      with the complete _evals parameter (no filtering applied).
    """
    # Create TrustworthyRAG instance
    tlm_rag = TrustworthyRAG(api_key=trustworthy_rag_api_key)

    # Store the original evals to verify they're passed through
    original_evals = tlm_rag._evals.copy()

    # Mock _is_tool_call_response to return False (non-tool call)
    with (
        mock.patch("cleanlab_tlm.internal.rag._is_tool_call_response", return_value=False),
        mock.patch("cleanlab_tlm.internal.api.api.tlm_rag_score") as mock_api_score,
    ):
        # Configure the mock to return a valid response
        mock_api_score.return_value = {eval_name: {"score": 0.8, "reason": "test"} for eval_name in original_evals}

        response = tlm_rag.score(
            query=test_query,
            context=test_context,
            response=test_response,
        )

        # Verify the API was called with the full evals (no filtering)
        assert mock_api_score.call_count == 1
        call_args = mock_api_score.call_args
        assert call_args is not None

        # Check that evals parameter matches the original evals
        called_evals = call_args.kwargs.get("evals")
        assert called_evals == original_evals

    # Should get a normal response
    assert isinstance(response, dict)
    for eval_dict in response.values():
        assert isinstance(eval_dict["score"], float)


def test_ordering_preserved_for_non_tool_calls(trustworthy_rag_api_key: str) -> None:  # noqa: F811
    """When not a tool call, ordering should match exactly what the mocked api.tlm_rag_score returns."""
    tlm_rag = TrustworthyRAG(api_key=trustworthy_rag_api_key)

    # Construct a mocked backend result with a specific insertion order
    mocked_backend = {
        "trustworthiness": {"score": 0.91},
        # Put evals in a custom order to ensure we preserve this order
        "query_ease": {"score": 0.11},
        "context_sufficiency": {"score": 0.22},
        "response_helpfulness": {"score": 0.33},
        "response_groundedness": {"score": 0.44},
    }

    with (
        mock.patch("cleanlab_tlm.internal.rag._is_tool_call_response", return_value=False),
        mock.patch("cleanlab_tlm.internal.api.api.tlm_rag_score", return_value=mocked_backend),
    ):
        result = tlm_rag.score(
            query=test_query,
            context=test_context,
            response=test_response,
        )

    assert isinstance(result, dict)
    assert list(result.keys()) == list(mocked_backend.keys())


def test_ordering_rebuilt_for_tool_calls(trustworthy_rag_api_key: str) -> None:  # noqa: F811
    """For tool calls, non-eval keys keep backend order, then all evals in self._evals order with filtered as None."""
    tlm_rag = TrustworthyRAG(api_key=trustworthy_rag_api_key)

    # Default eval order from TrustworthyRAG
    eval_order = [e.name for e in tlm_rag._evals]
    assert eval_order == [
        "context_sufficiency",
        "response_groundedness",
        "response_helpfulness",
        "query_ease",
    ]

    # Backend only processes non-response evals during tool-calls (decorator filters response-based evals)
    # Intentionally put processed evals in a non-evals order to ensure rebuild will override to eval_order
    mocked_backend_processed = {
        "trustworthiness": {"score": 0.9},
        "query_ease": {"score": 0.5},
        "context_sufficiency": {"score": 0.8},
    }

    with (
        mock.patch("cleanlab_tlm.internal.rag._is_tool_call_response", return_value=True),
        mock.patch("cleanlab_tlm.internal.api.api.tlm_rag_score", return_value=mocked_backend_processed),
    ):
        result = tlm_rag.score(
            query=test_query,
            context=test_context,
            response=test_response,
            prompt=test_prompt,
        )

    assert isinstance(result, dict)

    # Non-eval keys (trustworthiness) should appear first preserving backend order
    expected_keys = ["trustworthiness", *eval_order]
    assert list(result.keys()) == expected_keys

    # Filtered response-based evals should be present with None score
    assert result["response_groundedness"]["score"] is None
    assert result["response_helpfulness"]["score"] is None
    assert result["query_ease"]["score"] == mocked_backend_processed["query_ease"]["score"]
    assert result["context_sufficiency"]["score"] == mocked_backend_processed["context_sufficiency"]["score"]

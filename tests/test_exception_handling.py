from typing import Any, cast

from cleanlab_tlm.errors import APITimeoutError
from cleanlab_tlm.internal.exception_handling import _handle_exception
from cleanlab_tlm.utils.rag import Eval


class TestExceptionHandling:
    def test_trustworthy_rag_response_exception_handling(self) -> None:
        """Test exception handling for TrustworthyRAGResponse with custom evals."""
        # Create custom evals
        custom_evals = [
            Eval(
                name="custom_eval_1",
                criteria="Custom evaluation criteria 1",
                query_identifier="query",
                context_identifier="context",
                response_identifier="response",
            ),
            Eval(
                name="custom_eval_2",
                criteria="Custom evaluation criteria 2",
                query_identifier="query",
                context_identifier="context",
                response_identifier="response",
            ),
            Eval(
                name="trustworthiness",
                criteria="Trustworthiness evaluation",
                query_identifier="query",
                context_identifier="context",
                response_identifier="response",
            ),  # Should be handled specially
        ]

        # Create an exception
        error = APITimeoutError("Test timeout error")

        # Handle the exception
        result = _handle_exception(
            e=error,
            capture_exceptions=True,
            batch_index=0,
            retryable=True,
            response_type="TrustworthyRAGResponse",
            evals=custom_evals,
        )

        # Verify the result
        assert result is not None
        assert isinstance(result, dict)
        assert "response" in result
        assert result["response"] is None

        # Check trustworthiness field
        assert "trustworthiness" in result
        trustworthiness = cast(dict[str, Any], result["trustworthiness"])
        assert "score" in trustworthiness
        assert trustworthiness["score"] is None
        assert "log" in trustworthiness
        assert "error" in trustworthiness["log"]
        assert "message" in trustworthiness["log"]["error"]
        assert "retryable" in trustworthiness["log"]["error"]
        assert trustworthiness["log"]["error"]["message"] == "Test timeout error"
        assert trustworthiness["log"]["error"]["retryable"] is True

        # Check custom eval fields
        assert "custom_eval_1" in result
        assert "score" in result["custom_eval_1"]
        assert result["custom_eval_1"]["score"] is None

        assert "custom_eval_2" in result
        assert "score" in result["custom_eval_2"]
        assert result["custom_eval_2"]["score"] is None

    def test_trustworthy_rag_score_exception_handling(self) -> None:
        """Test exception handling for TrustworthyRAGScore with custom evals."""
        # Create custom evals
        custom_evals = [
            Eval(
                name="custom_eval_1",
                criteria="Custom evaluation criteria 1",
                query_identifier="query",
                context_identifier="context",
                response_identifier="response",
            ),
            Eval(
                name="custom_eval_2",
                criteria="Custom evaluation criteria 2",
                query_identifier="query",
                context_identifier="context",
                response_identifier="response",
            ),
            Eval(
                name="trustworthiness",
                criteria="Trustworthiness evaluation",
                query_identifier="query",
                context_identifier="context",
                response_identifier="response",
            ),  # Should be handled specially
        ]

        # Create an exception
        error = APITimeoutError("Test timeout error")

        # Handle the exception
        result = _handle_exception(
            e=error,
            capture_exceptions=True,
            batch_index=0,
            retryable=True,
            response_type="TrustworthyRAGScore",
            evals=custom_evals,
        )

        # Verify the result
        assert result is not None
        assert isinstance(result, dict)
        assert "response" not in result

        # Check trustworthiness field
        assert "trustworthiness" in result
        trustworthiness = cast(dict[str, Any], result["trustworthiness"])
        assert "score" in trustworthiness
        assert trustworthiness["score"] is None
        assert "log" in trustworthiness
        assert "error" in trustworthiness["log"]
        assert "message" in trustworthiness["log"]["error"]
        assert "retryable" in trustworthiness["log"]["error"]
        assert trustworthiness["log"]["error"]["message"] == "Test timeout error"
        assert trustworthiness["log"]["error"]["retryable"] is True

        # Check custom eval fields
        assert "custom_eval_1" in result
        assert "score" in result["custom_eval_1"]
        assert result["custom_eval_1"]["score"] is None

        assert "custom_eval_2" in result
        assert "score" in result["custom_eval_2"]
        assert result["custom_eval_2"]["score"] is None

    def test_trustworthy_rag_response_exception_handling_no_evals(self) -> None:
        """Test exception handling for TrustworthyRAGResponse with no evals."""
        # Create an exception
        error = APITimeoutError("Test timeout error")

        expected_result_length = 2

        # Handle the exception
        result = _handle_exception(
            e=error,
            capture_exceptions=True,
            batch_index=0,
            retryable=True,
            response_type="TrustworthyRAGResponse",
            evals=None,
        )

        # Verify the result
        assert result is not None
        assert isinstance(result, dict)
        assert "response" in result
        assert result["response"] is None

        # Check trustworthiness field
        assert "trustworthiness" in result
        trustworthiness = cast(dict[str, Any], result["trustworthiness"])
        assert "score" in trustworthiness
        assert trustworthiness["score"] is None
        assert "log" in trustworthiness
        assert "error" in trustworthiness["log"]

        # Should only have response and trustworthiness fields
        assert len(result) == expected_result_length

    def test_trustworthy_rag_score_exception_handling_no_evals(self) -> None:
        """Test exception handling for TrustworthyRAGScore with no evals."""
        # Create an exception
        error = APITimeoutError("Test timeout error")

        # Handle the exception
        result = _handle_exception(
            e=error,
            capture_exceptions=True,
            batch_index=0,
            retryable=True,
            response_type="TrustworthyRAGScore",
            evals=None,
        )

        # Verify the result
        assert result is not None
        assert isinstance(result, dict)
        assert "response" not in result

        # Check trustworthiness field
        assert "trustworthiness" in result
        trustworthiness = cast(dict[str, Any], result["trustworthiness"])
        assert "score" in trustworthiness
        assert trustworthiness["score"] is None
        assert "log" in trustworthiness
        assert "error" in trustworthiness["log"]

        # Should only have trustworthiness field
        assert len(result) == 1

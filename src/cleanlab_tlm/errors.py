class HandledError(Exception):
    pass


class ValidationError(HandledError):
    pass


class APIError(Exception):
    pass


class APITimeoutError(HandledError):
    pass


class InvalidProjectConfiguration(HandledError):
    pass


class RateLimitError(HandledError):
    def __init__(self, message: str, retry_after: int):
        self.message = message
        self.retry_after = retry_after


class TlmBadRequest(HandledError):
    def __init__(self, message: str, retryable: bool):
        self.message = message
        self.retryable = retryable


class TlmServerError(APIError):
    def __init__(self, message: str, status_code: int) -> None:
        self.message = message
        self.status_code = status_code


class TlmPartialSuccess(APIError):
    """TLM request partially succeeded. Still returns result to user."""

    pass


class TlmNotCalibratedError(HandledError):
    pass

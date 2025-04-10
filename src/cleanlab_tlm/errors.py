class HandledError(Exception):
    pass


class ValidationError(HandledError):
    pass


class APIError(Exception):
    pass


class HTTPBadRequestError(APIError):
    pass


class AuthError(HandledError):
    pass


class APITimeoutError(HandledError):
    pass


class InvalidProjectConfigurationError(HandledError):
    pass


class MissingApiKeyError(ValueError):
    def __str__(self) -> str:
        return "No API key provided. Please provide an API key using the `api_key` argument (`TLM(api_key=...)`) or by setting the `CLEANLAB_TLM_API_KEY` environment variable."


class RateLimitError(HandledError):
    def __init__(self, message: str, retry_after: int):
        self.message = message
        self.retry_after = retry_after


class TlmBadRequestError(HandledError):
    def __init__(self, message: str, retryable: bool):
        self.message = message
        self.retryable = retryable


class TlmServerError(APIError):
    def __init__(self, message: str, status_code: int) -> None:
        self.message = message
        self.status_code = status_code


class TlmPartialSuccessError(APIError):
    """TLM request partially succeeded. Still returns result to user."""


class TlmNotCalibratedError(HandledError):
    pass


# HTTP status codes
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_UNPROCESSABLE_ENTITY = 422
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVICE_UNAVAILABLE = 503

from __future__ import annotations

import asyncio
import os
import ssl
import time
import warnings
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

import aiohttp
import aiohttp.client_exceptions
from tqdm import tqdm

from cleanlab_tlm.errors import (
    HTTP_OK,
    HTTP_TOO_MANY_REQUESTS,
    HTTP_UNPROCESSABLE_ENTITY,
    APIError,
    InvalidProjectConfigurationError,
    RateLimitError,
    TlmBadRequestError,
    TlmPartialSuccessError,
    TlmServerError,
)
from cleanlab_tlm.internal.types import JSONDict

if TYPE_CHECKING:
    import requests

    from cleanlab_tlm.internal.concurrency import TlmRateHandler


base_url = os.environ.get("CLEANLAB_API_BASE_URL", "https://api.cleanlab.ai/api")
tlm_base_url = f"{base_url}/v0/trustworthy_llm"


def _construct_headers(api_key: Optional[str], content_type: Optional[str] = "application/json") -> JSONDict:
    retval = {}
    if api_key:
        retval["Authorization"] = f"bearer {api_key}"
    if content_type:
        retval["Content-Type"] = content_type
    retval["Client-Type"] = "python-api"
    retval["X-Tlm-Origin"] = "standalone"
    return retval


def handle_api_error(res: requests.Response) -> None:
    handle_api_error_from_json(res.json(), res.status_code)


def handle_api_error_from_json(res_json: JSONDict, status_code: Optional[int] = None) -> None:
    if "code" in res_json and "description" in res_json:  # AuthError or UserQuotaError format
        if res_json["code"] == "user_soft_quota_exceeded":
            pass  # soft quota limit is going away soon, so ignore it
        else:
            raise APIError(res_json["description"])

    if isinstance(res_json, dict) and res_json.get("error", None) is not None:
        error = res_json["error"]
        if (
            status_code == HTTP_UNPROCESSABLE_ENTITY
            and isinstance(error, dict)
            and error.get("code", None) == "UNSUPPORTED_PROJECT_CONFIGURATION"
        ):
            raise InvalidProjectConfigurationError(error["description"])
        raise APIError(res_json["error"])

    if status_code != HTTP_OK:
        raise APIError(f"API call failed with status code {status_code}")


def handle_rate_limit_error_from_resp(resp: aiohttp.ClientResponse) -> None:
    """Catches 429 (rate limit) errors."""
    if resp.status == HTTP_TOO_MANY_REQUESTS:
        raise RateLimitError(
            f"Rate limit exceeded on {resp.url}",
            int(resp.headers.get("Retry-After", 0)),
        )


async def handle_tlm_client_error_from_resp(resp: aiohttp.ClientResponse, batch_index: Optional[int]) -> None:
    """Catches 4XX (client error) errors."""
    if 400 <= resp.status < 500:  # noqa: PLR2004
        try:
            res_json = await resp.json()
            error_message = res_json["error"]
            retryable = False
        except Exception:
            error_message = (
                "TLM query failed. Please try again and contact support@cleanlab.ai if the problem persists."
            )
            retryable = True
        if batch_index is not None:
            error_message = f"Error executing query at index {batch_index}:\n{error_message}"

        raise TlmBadRequestError(error_message, retryable)


async def handle_tlm_api_error_from_resp(resp: aiohttp.ClientResponse, batch_index: Optional[int]) -> None:
    """Catches 5XX (server error) errors."""
    if 500 <= resp.status < 600:  # noqa: PLR2004
        try:
            res_json = await resp.json()
            error_message = res_json["error"]
        except Exception:
            error_message = (
                "TLM query failed. Please try again and contact support@cleanlab.ai if the problem persists."
            )

        if batch_index is not None:
            error_message = f"Error executing query at index {batch_index}:\n{error_message}"

        raise TlmServerError(error_message, resp.status)


def poll_progress(progress_id: str, request_function: Callable[[str], JSONDict], description: str) -> JSONDict:
    with tqdm(total=1, desc=description, bar_format="{desc}: {percentage:3.0f}%|{bar}|") as pbar:
        res = request_function(progress_id)
        while res["status"] != "complete":
            if res["status"] == "error":
                raise APIError(res["error_message"])
            pbar.update(float(res["progress"]) - pbar.n)
            time.sleep(0.5)
            res = request_function(progress_id)
        pbar.update(float(1) - pbar.n)
    return res


def tlm_retry(func: Callable[..., Any]) -> Callable[..., Any]:
    """Implements TLM retry decorator, with special handling for rate limit retries."""

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # total number of tries = number of retries + original try
        max_general_retries = kwargs.pop("retries", 0)
        max_connection_error_retries = 20

        sleep_time = 0
        error_message = ""

        num_general_retry = 0
        num_connection_error_retry = 0

        while num_general_retry <= max_general_retries and num_connection_error_retry <= max_connection_error_retries:
            await asyncio.sleep(sleep_time)
            try:
                return await func(*args, **kwargs)
            except ssl.SSLCertVerificationError:
                warnings.warn(
                    "Please ensure that your SSL certificates are up to date. If you installed python via python pkg installer, please make sure to execute 'Install Certificates.command' in the python installation directory."
                )
                raise
            except aiohttp.client_exceptions.ClientConnectorError as e:
                if num_connection_error_retry == (max_connection_error_retries // 2):
                    warnings.warn(f"Connection error after {num_connection_error_retry} retries. Retrying...")
                sleep_time = min(2**num_connection_error_retry, 60)
                # note: we have a different counter for connection errors, because we want to retry connection errors more times
                num_connection_error_retry += 1
                error_message = str(e)
            except RateLimitError as e:
                # note: we don't increment num_general_retry here, because we don't want rate limit retries to count against the total number of retries
                sleep_time = e.retry_after
            except TlmBadRequestError:
                # dont retry for client-side errors
                raise
            except Exception as e:
                sleep_time = 2**num_general_retry
                num_general_retry += 1
                error_message = str(e)

        if num_connection_error_retry > max_connection_error_retries:
            raise APIError(
                f"Connection error after {num_connection_error_retry} retries. {error_message}",
                -1,
            )

        raise APIError(f"TLM failed after {num_general_retry} attempts. {error_message}", -1)

    return wrapper


@tlm_retry
async def tlm_prompt(
    api_key: str,
    prompt: str,
    quality_preset: str,
    task: str,
    options: Optional[JSONDict],
    rate_handler: TlmRateHandler,
    client_session: Optional[aiohttp.ClientSession] = None,
    batch_index: Optional[int] = None,
    constrain_outputs: Optional[list[str]] = None,
) -> JSONDict:
    """
    Prompt Trustworthy Language Model with a question, and get back its answer along with a confidence score

    Args:
        api_key (str): API key for auth
        prompt (str): prompt for TLM to respond to
        quality_preset (str): quality preset to use to generate response
        task (str): task type for evaluation
        options (JSONDict): additional parameters for TLM
        rate_handler (TlmRateHandler): concurrency handler used to manage TLM request rate
        client_session (aiohttp.ClientSession): client session used to issue TLM request
        batch_index (Optional[int], optional): index of prompt in batch, used for error messages. Defaults to None if not in batch.
        constrain_outputs (Optional[List[str]], optional): list of strings to constrain the output of the TLM to. Defaults to None.
    Returns:
        JSONDict: dictionary with TLM response and confidence score
    """
    local_scoped_client = False
    if not client_session:
        client_session = aiohttp.ClientSession()
        local_scoped_client = True

    try:
        async with rate_handler:
            base_api_url = os.environ.get("CLEANLAB_API_TLM_BASE_URL", tlm_base_url)
            res = await client_session.post(
                f"{base_api_url}/prompt",
                json={
                    "prompt": prompt,
                    "quality": quality_preset,
                    "task": task,
                    "options": options or {},
                    "user_id": api_key,
                    "client_id": api_key,
                    "constrain_outputs": constrain_outputs,
                },
                headers=_construct_headers(api_key),
            )

            res_json = await res.json()

            handle_rate_limit_error_from_resp(res)
            await handle_tlm_client_error_from_resp(res, batch_index)
            await handle_tlm_api_error_from_resp(res, batch_index)

            if not res_json.get("deberta_success", True):
                raise TlmPartialSuccessError("Partial failure on deberta call -- slowdown request rate.")

    finally:
        if local_scoped_client:
            await client_session.close()

    return cast(JSONDict, res_json)


@tlm_retry
async def tlm_get_confidence_score(
    api_key: str,
    prompt: str,
    response: dict[str, Any],
    quality_preset: str,
    task: str,
    options: Optional[JSONDict],
    rate_handler: TlmRateHandler,
    client_session: Optional[aiohttp.ClientSession] = None,
    batch_index: Optional[int] = None,
) -> JSONDict:
    """
    Query Trustworthy Language Model for a confidence score for the prompt-response pair.

    Args:
        api_key (str): API key for auth
        prompt (str): prompt for TLM to get confidence score for
        response (Dict[str, Any]): dictionary containing response and optional metadata
        quality_preset (str): quality preset to use to generate confidence score
        task (str): task type for evaluation
        options (JSONDict): additional parameters for TLM
        rate_handler (TlmRateHandler): concurrency handler used to manage TLM request rate
        client_session (aiohttp.ClientSession): client session used to issue TLM request
        batch_index (Optional[int], optional): index of prompt in batch, used for error messages. Defaults to None if not in batch.

    Returns:
        JSONDict: dictionary with TLM confidence score
    """
    local_scoped_client = False
    if not client_session:
        client_session = aiohttp.ClientSession()
        local_scoped_client = True

    try:
        async with rate_handler:
            res = await client_session.post(
                f"{tlm_base_url}/get_confidence_score",
                json={
                    "prompt": prompt,
                    "response": response,
                    "quality": quality_preset,
                    "task": task,
                    "options": options or {},
                },
                headers=_construct_headers(api_key),
            )

            res_json = await res.json()

            handle_rate_limit_error_from_resp(res)
            await handle_tlm_client_error_from_resp(res, batch_index)
            await handle_tlm_api_error_from_resp(res, batch_index)

    finally:
        if local_scoped_client:
            await client_session.close()

    return cast(JSONDict, res_json)

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
    HTTP_BAD_REQUEST,
    HTTP_OK,
    HTTP_TOO_MANY_REQUESTS,
    HTTP_UNAUTHORIZED,
    HTTP_UNPROCESSABLE_ENTITY,
    APIError,
    AuthError,
    HTTPBadRequestError,
    InvalidProjectConfigurationError,
    RateLimitError,
    TlmBadRequestError,
    TlmPartialSuccessError,
    TlmServerError,
)
from cleanlab_tlm.internal.constants import (
    _TLM_CLIENT_ID_KEY,
    _TLM_CONSTRAIN_OUTPUTS_KEY,
    _TLM_CONTEXT_KEY,
    _TLM_DEBERTA_SUCCESS_KEY,
    _TLM_EVALS_KEY,
    _TLM_OPTIONS_KEY,
    _TLM_PROMPT_KEY,
    _TLM_QUALITY_KEY,
    _TLM_QUERY_KEY,
    _TLM_RESPONSE_KEY,
    _TLM_TASK_KEY,
    _TLM_TRUSTWORTHINESS_KEY,
    _TLM_USER_ID_KEY,
)
from cleanlab_tlm.internal.types import JSONDict

if TYPE_CHECKING:
    import requests

    from cleanlab_tlm.internal.concurrency import TlmRateHandler
    from cleanlab_tlm.utils.rag import Eval


base_url = os.environ.get("CLEANLAB_API_BASE_URL", "https://api.cleanlab.ai/api")
tlm_base_url = f"{base_url}/v0/trustworthy_llm"
tlm_rag_base_url = f"{base_url}/v1/rag_trustworthy_llm"


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


async def handle_http_bad_request_error_from_resp(resp: aiohttp.ClientResponse) -> None:
    """Catches 400 (bad request) errors."""
    if resp.status == HTTP_BAD_REQUEST:
        res_json = await resp.json()
        raise HTTPBadRequestError(res_json["error"])


async def handle_api_key_error_from_resp(resp: aiohttp.ClientResponse) -> None:
    """Catches 401 (unauthorized) errors."""
    if resp.status == HTTP_UNAUTHORIZED:
        res_json = await resp.json()
        if res_json.get("code", None) == "invalid_api_key":
            raise AuthError("Invalid API key. Check https://tlm.cleanlab.ai/ for your current API key.")


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
            except AuthError:
                # dont retry for auth errors
                raise
            except (HTTPBadRequestError, Exception) as e:
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
                    _TLM_PROMPT_KEY: prompt,
                    _TLM_QUALITY_KEY: quality_preset,
                    _TLM_TASK_KEY: task,
                    _TLM_OPTIONS_KEY: options or {},
                    _TLM_USER_ID_KEY: api_key,
                    _TLM_CLIENT_ID_KEY: api_key,
                    _TLM_CONSTRAIN_OUTPUTS_KEY: constrain_outputs,
                },
                headers=_construct_headers(api_key),
            )

            res_json = await res.json()

            await handle_api_key_error_from_resp(res)
            await handle_http_bad_request_error_from_resp(res)
            handle_rate_limit_error_from_resp(res)
            await handle_tlm_client_error_from_resp(res, batch_index)
            await handle_tlm_api_error_from_resp(res, batch_index)

            if not res_json.get(_TLM_DEBERTA_SUCCESS_KEY, True):
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
                    _TLM_PROMPT_KEY: prompt,
                    _TLM_RESPONSE_KEY: response,
                    _TLM_QUALITY_KEY: quality_preset,
                    _TLM_TASK_KEY: task,
                    _TLM_OPTIONS_KEY: options or {},
                },
                headers=_construct_headers(api_key),
            )

            res_json = await res.json()

            await handle_api_key_error_from_resp(res)
            await handle_http_bad_request_error_from_resp(res)
            handle_rate_limit_error_from_resp(res)
            await handle_tlm_client_error_from_resp(res, batch_index)
            await handle_tlm_api_error_from_resp(res, batch_index)

            if not res_json.get(_TLM_DEBERTA_SUCCESS_KEY, True):
                raise TlmPartialSuccessError("Partial failure on deberta call -- slowdown request rate.")

    finally:
        if local_scoped_client:
            await client_session.close()

    return cast(JSONDict, res_json)


@tlm_retry
async def tlm_rag_generate(
    api_key: str,
    prompt: str,
    query: str,
    context: str,
    evals: list[Eval],
    quality_preset: str,
    options: Optional[JSONDict],
    rate_handler: TlmRateHandler,
    client_session: Optional[aiohttp.ClientSession] = None,
    batch_index: Optional[int] = None,
    constrain_outputs: Optional[list[str]] = None,
) -> JSONDict:
    """
    Generate a response using Trustworthy Language Model with RAG (Retrieval-Augmented Generation) capabilities

    Args:
        api_key (str): API key for auth
        prompt (str): prompt for TLM to respond to
        quality_preset (str): quality preset to use to generate response
        options (JSONDict): additional parameters for TLM
        rate_handler (TlmRateHandler): concurrency handler used to manage TLM request rate
        client_session (aiohttp.ClientSession): client session used to issue TLM request
        batch_index (Optional[int], optional): index of prompt in batch, used for error messages. Defaults to None if not in batch.
        constrain_outputs (Optional[list[str]], optional): list of strings to constrain the output of the TLM to. Defaults to None.
        query (Optional[str], optional): query for RAG context retrieval. Defaults to None.
        context (Optional[str], optional): context information for RAG. Defaults to None.
        evals (Optional[list[dict[str, Union[str, dict[str, str]]]]], optional): list of evaluation criteria. Defaults to None.
    Returns:
        JSONDict: dictionary with TLM response, trustworthiness score, and any evaluation results
    """
    local_scoped_client = False
    if not client_session:
        client_session = aiohttp.ClientSession()
        local_scoped_client = True

    try:
        async with rate_handler:
            res = await client_session.post(
                f"{tlm_rag_base_url}/generate",
                json={
                    _TLM_PROMPT_KEY: prompt,
                    _TLM_QUERY_KEY: query,
                    _TLM_CONTEXT_KEY: context,
                    _TLM_EVALS_KEY: [evaluation.__dict__ for evaluation in evals] if evals else [],
                    _TLM_QUALITY_KEY: quality_preset,
                    _TLM_OPTIONS_KEY: options or {},
                    _TLM_USER_ID_KEY: api_key,
                    _TLM_CLIENT_ID_KEY: api_key,
                    _TLM_CONSTRAIN_OUTPUTS_KEY: constrain_outputs,
                },
                headers=_construct_headers(api_key),
            )

            res_json = await res.json()

            await handle_api_key_error_from_resp(res)
            await handle_http_bad_request_error_from_resp(res)
            handle_rate_limit_error_from_resp(res)
            await handle_tlm_client_error_from_resp(res, batch_index)
            await handle_tlm_api_error_from_resp(res, batch_index)

            if not res_json.get(_TLM_DEBERTA_SUCCESS_KEY, True):
                raise TlmPartialSuccessError("Partial failure on deberta call -- slowdown request rate.")

    finally:
        if local_scoped_client:
            await client_session.close()

    # Create a dictionary with the specified key order
    ordered_res = {}

    if _TLM_RESPONSE_KEY in res_json:
        ordered_res[_TLM_RESPONSE_KEY] = res_json[_TLM_RESPONSE_KEY]

    if _TLM_TRUSTWORTHINESS_KEY in res_json:
        ordered_res[_TLM_TRUSTWORTHINESS_KEY] = res_json[_TLM_TRUSTWORTHINESS_KEY]

    # Add any eval-related keys in their original order
    for evaluation in evals:
        if evaluation.name not in [_TLM_RESPONSE_KEY, _TLM_TRUSTWORTHINESS_KEY]:
            ordered_res[evaluation.name] = res_json[evaluation.name]

    return ordered_res


@tlm_retry
async def tlm_rag_score(
    api_key: str,
    response: dict[str, Any],
    prompt: str,
    query: str,
    context: str,
    evals: list[Eval],
    quality_preset: str,
    options: Optional[JSONDict],
    rate_handler: TlmRateHandler,
    client_session: Optional[aiohttp.ClientSession] = None,
    batch_index: Optional[int] = None,
    constrain_outputs: Optional[list[str]] = None,
) -> JSONDict:
    """
    Score a response using Trustworthy Language Model with RAG (Retrieval-Augmented Generation) evaluation

    Args:
        api_key (str): API key for auth
        response (str): response to be evaluated
        prompt (str): prompt that was used to generate the response
        quality_preset (str): quality preset to use for evaluation
        options (JSONDict): additional parameters for TLM
        rate_handler (TlmRateHandler): concurrency handler used to manage TLM request rate
        client_session (aiohttp.ClientSession): client session used to issue TLM request
        batch_index (Optional[int], optional): index of prompt in batch, used for error messages. Defaults to None if not in batch.
        constrain_outputs (Optional[list[str]], optional): list of strings to constrain the output of the TLM to. Defaults to None.
        query (Optional[str], optional): query used for RAG context retrieval. Defaults to None.
        context (Optional[str], optional): context information used for RAG. Defaults to None.
        evals (Optional[list[Eval]], optional): list of evaluation criteria objects. Defaults to None.
    Returns:
        JSONDict: dictionary with trustworthiness score and any evaluation results
    """
    local_scoped_client = False
    if not client_session:
        client_session = aiohttp.ClientSession()
        local_scoped_client = True

    try:
        async with rate_handler:
            res = await client_session.post(
                f"{tlm_rag_base_url}/score",
                json={
                    _TLM_RESPONSE_KEY: response,
                    _TLM_PROMPT_KEY: prompt,
                    _TLM_QUERY_KEY: query,
                    _TLM_CONTEXT_KEY: context,
                    _TLM_EVALS_KEY: [evaluation.__dict__ for evaluation in evals] if evals else [],
                    _TLM_QUALITY_KEY: quality_preset,
                    _TLM_OPTIONS_KEY: options or {},
                    _TLM_USER_ID_KEY: api_key,
                    _TLM_CLIENT_ID_KEY: api_key,
                    _TLM_CONSTRAIN_OUTPUTS_KEY: constrain_outputs,
                },
                headers=_construct_headers(api_key),
            )

            res_json = await res.json()

            await handle_api_key_error_from_resp(res)
            await handle_http_bad_request_error_from_resp(res)
            handle_rate_limit_error_from_resp(res)
            await handle_tlm_client_error_from_resp(res, batch_index)
            await handle_tlm_api_error_from_resp(res, batch_index)

            if not res_json.get(_TLM_DEBERTA_SUCCESS_KEY, True):
                raise TlmPartialSuccessError("Partial failure on deberta call -- slowdown request rate.")

    finally:
        if local_scoped_client:
            await client_session.close()

    # Create a dictionary with the specified key order
    ordered_res = {}

    if _TLM_RESPONSE_KEY in res_json:
        ordered_res[_TLM_RESPONSE_KEY] = res_json[_TLM_RESPONSE_KEY]

    if _TLM_TRUSTWORTHINESS_KEY in res_json:
        ordered_res[_TLM_TRUSTWORTHINESS_KEY] = res_json[_TLM_TRUSTWORTHINESS_KEY]

    # Add any eval-related keys in their original order
    for evaluation in evals:
        if evaluation.name not in [_TLM_RESPONSE_KEY, _TLM_TRUSTWORTHINESS_KEY]:
            ordered_res[evaluation.name] = res_json[evaluation.name]

    return ordered_res

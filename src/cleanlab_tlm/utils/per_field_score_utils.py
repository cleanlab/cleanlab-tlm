import json
from typing import TYPE_CHECKING, Optional, Union

from cleanlab_tlm.tlm import TLMScore

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion


def _get_untrustworthy_fields(
    response: Optional["ChatCompletion"],
    tlm_result: Union[TLMScore, "ChatCompletion"],
    threshold: float,
    display_details: bool,
    class_name: str,
) -> list[str]:
    try:
        from openai.types.chat import ChatCompletion
    except ImportError as e:
        raise ImportError(
            f"OpenAI is required to use the {class_name} class. Please install it with `pip install openai`."
        ) from e

    if isinstance(tlm_result, dict):
        if response is None:
            raise ValueError("'response' is required when tlm_result is a TLMScore object")

        tlm_metadata = tlm_result
        response_text = response.choices[0].message.content or "{}"

    elif isinstance(tlm_result, ChatCompletion):
        if getattr(tlm_result, "tlm_metadata", None) is None:
            raise ValueError("tlm_result must contain tlm_metadata.")

        tlm_metadata = tlm_result.tlm_metadata  # type: ignore
        response_text = tlm_result.choices[0].message.content or "{}"

    else:
        raise TypeError("tlm_result must be a TLMScore or ChatCompletion object.")

    if "per_field_score" not in tlm_metadata.get("log", {}):
        raise ValueError(
            "`per_field_score` is not present in the log.\n"
            "`get_untrustworthy_fields()` can only be called scoring structured outputs responses and specifying "
            "`per_field_score` in the `log` option for TLM."
        )

    try:
        so_response = json.loads(response_text)
    except Exception:
        raise ValueError(
            "The LLM response must be a valid JSON output (use `response_format` to specify the output format)"
        )

    per_field_score = tlm_metadata["log"]["per_field_score"]
    per_score_details = []

    for key, value in per_field_score.items():
        score = value["score"]
        if float(score) < threshold:
            key_details = {
                "response": so_response[key],
                "score": score,
                "explanation": value["explanation"],
            }
            per_score_details.append({key: key_details})

    per_score_details.sort(key=lambda x: next(iter(x.values()))["score"])
    untrustworthy_fields = [next(iter(item.keys())) for item in per_score_details]

    if display_details:
        if len(untrustworthy_fields) == 0:
            print("No untrustworthy fields found")

        else:
            print(f"Untrustworthy fields: {untrustworthy_fields}\n")
            for item in per_score_details:
                print(f"Field: {next(iter(item.keys()))}")
                details = next(iter(item.values()))
                print(f"Response: {details['response']}")
                print(f"Score: {details['score']}")
                print(f"Explanation: {details['explanation']}")
                print()

    return untrustworthy_fields

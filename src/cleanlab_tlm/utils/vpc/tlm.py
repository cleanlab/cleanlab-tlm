from typing_extensions import NotRequired, TypedDict

from cleanlab_tlm.tlm import TLMOptions


class ModelProvider(TypedDict):
    """Typed dict of model provider options for the Trustworthy Language Model."""

    api_base: NotRequired[str]
    api_key: NotRequired[str]
    api_version: NotRequired[str]
    provider: NotRequired[str]


class VPCTLMOptions(TLMOptions):
    model_provider: NotRequired[ModelProvider]

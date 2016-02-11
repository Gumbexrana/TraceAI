from collections.abc import Sequence
from typing import Any, Dict, List, Literal, TypedDict, Union

from typing_extensions import Required, TypeAlias

from fi_instrumentation.fi_types import (
    FiLLMProviderValues,
    FiLLMSystemValues,
    FiMimeTypeValues,
    FiSpanKindValues,
)

FiSpanKind = Union[
    Literal[
        "agent",
        "chain",
        "embedding",
        "evaluator",
        "guardrail",
        "llm",
        "reranker",
        "retriever",
        "tool",
        "unknown",
    ],
    FiSpanKindValues,
]
FiMimeType = Union[
    Literal["application/json", "text/plain"],
    FiMimeTypeValues,
]
FiLLMProvider: TypeAlias = Union[str, FiLLMProviderValues]
FiLLMSystem: TypeAlias = Union[str, FiLLMSystemValues]


class Image(TypedDict, total=False):
    url: str


class TextMessageContent(TypedDict):
    type: Literal["text"]
    text: str


class ImageMessageContent(TypedDict):
    type: Literal["image"]
    image: Image


MessageContent: TypeAlias = Union[TextMessageContent, ImageMessageContent]


class ToolCallFunction(TypedDict, total=False):
    name: str
    arguments: Union[str, Dict[str, Any]]


class ToolCall(TypedDict, total=False):
    id: str
    function: ToolCallFunction


class Message(TypedDict, total=False):
    role: str
    content: str
    contents: "Sequence[MessageContent]"
    tool_call_id: str

import logging
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping, Tuple

from fi_instrumentation.fi_types import (
    MessageAttributes,
    SpanAttributes,
    ToolCallAttributes,
)
from opentelemetry.util.types import AttributeValue

if TYPE_CHECKING:
    from mistralai.models import ChatCompletionResponse

__all__ = ("_ResponseAttributesExtractor",)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _ResponseAttributesExtractor:
    def get_attributes_from_response(
        self,
        response: Any,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield from _get_attributes_from_chat_completion_response(response)


def _get_attributes_from_chat_completion_response(
    response: "ChatCompletionResponse",
) -> Iterator[Tuple[str, AttributeValue]]:
    if model := getattr(response, "model", None):
        yield SpanAttributes.GEN_AI_REQUEST_MODEL, model
    if usage := getattr(response, "usage", None):
        yield from _get_attributes_from_completion_usage(usage)
    if (choices := getattr(response, "choices", None)) and isinstance(
        choices, Iterable
    ):
        for choice in choices:
            if (index := _get_attribute_or_value(choice, "index")) is None:
                continue
            if message := _get_attribute_or_value(choice, "message"):
                for key, value in _get_attributes_from_chat_completion_message(message):
                    yield f"{SpanAttributes.GEN_AI_OUTPUT_MESSAGES}.{index}.{key}", value


class _StreamResponseAttributesExtractor:
    def get_attributes_from_response(
        self,
        response: Any,

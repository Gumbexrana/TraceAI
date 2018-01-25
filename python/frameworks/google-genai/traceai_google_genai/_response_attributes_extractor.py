import logging
from typing import Any, Iterable, Iterator, Mapping, Tuple

from google.genai import types
from opentelemetry.util.types import AttributeValue

from fi_instrumentation import safe_json_dumps
from traceai_google_genai._utils import (
    _as_output_attributes,
    _io_value_and_type,
)
from fi_instrumentation.fi_types import MessageAttributes, SpanAttributes, ToolCallAttributes

__all__ = ("_ResponseAttributesExtractor",)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _ResponseAttributesExtractor:
    def get_attributes(self, response: Any) -> Iterator[Tuple[str, AttributeValue]]:
        yield from _as_output_attributes(
            _io_value_and_type(response),
        )

    def get_extra_attributes(
        self,
        response: Any,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield from self._get_attributes_from_generate_content(
            response=response,
            request_parameters=request_parameters,
        )

    def _get_attributes_from_generate_content(
        self,
        response: Any,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        # https://github.com/googleapis/python-genai/blob/e9e84aa38726e7b65796812684d9609461416b11/google/genai/types.py#L2981  # noqa: E501
        if model_version := getattr(response, "model_version", None):
            yield SpanAttributes.GEN_AI_REQUEST_MODEL, model_version
        if usage_metadata := getattr(response, "usage_metadata", None):
            yield from self._get_attributes_from_generate_content_usage(usage_metadata)
        if (candidates := getattr(response, "candidates", None)) and isinstance(
            candidates, Iterable
        ):
            index = -1
            for candidate in candidates:
                # TODO: This is a hack to get the index of the candidate.
                #       Might be a better way to do this.
                # Keep track of previous index to increment if index not found
                index = (
                    index + 1
                    if getattr(candidate, "index") is None
                    else getattr(candidate, "index")
                )
                if content := getattr(candidate, "content", None):
                    for key, value in self._get_attributes_from_generate_content_content(content):
                        yield f"{SpanAttributes.GEN_AI_OUTPUT_MESSAGES}.{index}.{key}", value

        # Handle automatic function calling history
        # For automatic function calling, the function call details are stored separately
        if automatic_history := getattr(response, "automatic_function_calling_history", None):
            yield from self._get_attributes_from_automatic_function_calling_history(
                automatic_history
            )

    def _get_attributes_from_generate_content_content(
        self,
        content: object,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if content_parts := getattr(content, "parts", None):
            yield from self._get_attributes_from_content_parts(content_parts)
        if role := getattr(content, "role", None):
            yield MessageAttributes.MESSAGE_ROLE, role

    def _get_attributes_from_content_parts(

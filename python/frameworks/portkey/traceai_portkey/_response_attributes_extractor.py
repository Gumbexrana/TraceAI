import logging
from typing import Any, Iterable, Iterator, Mapping, Tuple

from opentelemetry.util.types import AttributeValue

from traceai_portkey._utils import _as_output_attributes, _io_value_and_type
from fi_instrumentation.fi_types import MessageAttributes, SpanAttributes

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
        yield from self._get_attributes_from_chat_completion(
            completion=response,
            request_parameters=request_parameters,
        )

import logging
from typing import Any, Iterator, NamedTuple, Optional, Protocol, Tuple

from fi_instrumentation import safe_json_dumps
from fi_instrumentation.fi_types import FiMimeTypeValues, SpanAttributes
from opentelemetry import trace as trace_api
from opentelemetry.util.types import Attributes, AttributeValue
from traceai_anthropic._with_span import _WithSpan

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _ValueAndType(NamedTuple):
    value: str
    type: FiMimeTypeValues


class _HasAttributes(Protocol):
    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]: ...

    def get_extra_attributes(self) -> Iterator[Tuple[str, AttributeValue]]: ...


def _finish_tracing(
    with_span: _WithSpan,
    has_attributes: _HasAttributes,
    status: Optional[trace_api.Status] = None,
) -> None:
    try:
        attributes: Attributes = dict(has_attributes.get_attributes())
    except Exception as e:
        logger.exception("Failed to get attributes")
        attributes = None

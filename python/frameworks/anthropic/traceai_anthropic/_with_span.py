import logging
from typing import Any, Dict, Optional, Union

from opentelemetry import trace as trace_api
from opentelemetry.util.types import Attributes, AttributeValue

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _WithSpan:
    __slots__ = (
        "_span",
        "_is_finished",
    )

    def __init__(
        self,
        span: trace_api.Span,
    ) -> None:
        self._span = span
        try:
            self._is_finished = not self._span.is_recording()
        except Exception:
            logger.exception("Failed to check if span is recording")
            print("Error checking span recording status, defaulting to finished")
            self._is_finished = True

    @property
    def is_finished(self) -> bool:
        return self._is_finished

    def set_attribute(self, key: str, value: Any) -> None:
        self._span.set_attribute(key, value)

    def set_attributes(self, attributes: Dict[str, AttributeValue]) -> None:
        self._span.set_attributes(attributes)

    def record_exception(self, exception: Exception) -> None:
        if self._is_finished:

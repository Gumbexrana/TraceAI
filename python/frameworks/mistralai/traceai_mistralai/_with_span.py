import logging
from typing import Optional

from opentelemetry import trace as trace_api
from opentelemetry.util.types import Attributes

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _WithSpan:
    __slots__ = (
        "_span",
        "_context_attributes",
        "_extra_attributes",
        "_is_finished",
    )

    def __init__(
        self,
        span: trace_api.Span,
        context_attributes: Attributes = None,
        extra_attributes: Attributes = None,
    ) -> None:
        self._span = span
        self._context_attributes = context_attributes
        self._extra_attributes = extra_attributes
        try:
            self._is_finished = not self._span.is_recording()
        except Exception:
            logger.exception("Failed to check if span is recording")
            self._is_finished = True

    @property

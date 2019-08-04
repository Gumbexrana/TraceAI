from typing import Any, Dict, Optional

from opentelemetry import trace as trace_api


class _WithSpan:
    """Helper class to manage span lifecycle and attributes."""

    __slots__ = ("_span", "_context_attributes", "_extra_attributes")

    def __init__(
        self,
        span: trace_api.Span,
        context_attributes: Optional[Dict[str, Any]] = None,
        extra_attributes: Optional[Dict[str, Any]] = None,
    ) -> None:

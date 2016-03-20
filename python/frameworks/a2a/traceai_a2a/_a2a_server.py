"""
A2A Protocol server-side instrumentation — ASGI middleware.

This middleware extracts incoming W3C TraceContext headers (traceparent/tracestate)
from A2A HTTP requests and starts a child span in that distributed trace.

Usage with Starlette / FastAPI:
    from starlette.applications import Starlette
    from traceai_a2a import A2ATracingMiddleware

    app = Starlette()
    app.add_middleware(A2ATracingMiddleware, tracer_provider=trace_provider)

When an A2A orchestrator calls this agent, its trace_id propagates via the
traceparent header. This middleware picks it up and ensures the receiving
agent's spans appear as children in the same distributed trace.
"""

import logging
from typing import Any, Callable, Dict, Optional

from opentelemetry import propagate
from opentelemetry import trace as trace_api
from opentelemetry.trace import Status, StatusCode, Tracer, TracerProvider

from traceai_a2a._attributes import get_task_attributes
from traceai_a2a._semantic_conventions import (
    A2A_AGENT_URL,
    A2A_MESSAGE_ROLE,
    A2A_SPAN_KIND_SERVER,
    A2A_TASK_ID,
    A2A_TASK_STATE,
    TASK_STATE_COMPLETED,
)

try:
    from fi_instrumentation.fi_types import SpanAttributes
    _FI_AVAILABLE = True
except ImportError:
    _FI_AVAILABLE = False

logger = logging.getLogger(__name__)


class _ASGICarrier:
    """
    Read-only carrier that exposes ASGI scope headers as a dict-like object
    for use with OpenTelemetry's propagate.extract().

    ASGI headers are list of (bytes, bytes) tuples; we decode them on access.
    """

    def __init__(self, headers: list) -> None:
        self._headers: Dict[str, str] = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in headers
        }

    def __getitem__(self, key: str) -> str:
        return self._headers[key.lower()]

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key.lower() in self._headers

    def keys(self):
        return self._headers.keys()

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self._headers.get(key.lower(), default)



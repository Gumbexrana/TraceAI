"""
traceai-a2a: OpenTelemetry instrumentation for Google's Agent-to-Agent (A2A) Protocol.

Enables distributed tracing across multi-agent boundaries by propagating
W3C TraceContext headers automatically on every A2A call.

Basic usage:
    from traceai_a2a import A2AInstrumentor
    A2AInstrumentor().instrument(tracer_provider=trace_provider)

For server-side (receiving agent):
    from traceai_a2a import A2ATracingMiddleware
    app.add_middleware(A2ATracingMiddleware, tracer_provider=trace_provider)
"""

import logging
from typing import Any, Collection, Optional

import wrapt
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import TracerProvider

from traceai_a2a._a2a_client import A2AClientWrapper
from traceai_a2a._a2a_server import A2ATracingMiddleware
from traceai_a2a.package import _instruments
from traceai_a2a.version import __version__

logger = logging.getLogger(__name__)

__all__ = [
    "A2AInstrumentor",
    "A2ATracingMiddleware",
    "__version__",
]

# These are the methods we patch on A2AClient
_SEND_TASK_METHOD = "send_task"
_SEND_TASK_STREAMING_METHOD = "send_task_streaming"


class A2AInstrumentor(BaseInstrumentor):
    """
    OpenTelemetry instrumentor for the Google Agent-to-Agent (A2A) Protocol.

    Instruments A2AClient to:
    - Create an A2A_CLIENT span for every outbound agent call
    - Inject W3C TraceContext (traceparent/tracestate) into outbound HTTP headers
    - Record task ID, task state, agent URL, message role, and artifact type
    - Stitch distributed multi-agent traces into a single trace view

    Installation:
        pip install traceAI-a2a

    Usage:
        from fi_instrumentation import register
        from fi_instrumentation.fi_types import ProjectType
        from traceai_a2a import A2AInstrumentor

        trace_provider = register(project_type=ProjectType.OBSERVE, project_name="my_app")
        A2AInstrumentor().instrument(tracer_provider=trace_provider)

        # Now use A2AClient as normal — tracing is automatic
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider: Optional[TracerProvider] = kwargs.get("tracer_provider")
        if tracer_provider is None:
            tracer_provider = trace_api.get_tracer_provider()

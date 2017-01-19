"""Instrumentor for IBM BeeAI Framework.

This module provides functions to configure BeeAI's OpenInference-based
instrumentation to send traces to TraceAI.
"""

import os
from typing import Any, Dict, Optional

from opentelemetry import trace
from opentelemetry.trace import TracerProvider
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, BatchSpanProcessor
from opentelemetry.sdk.resources import Resource


class BeeAIInstrumentorWrapper:
    """Wrapper around BeeAI's OpenInference instrumentor.

    This class provides a convenient way to configure BeeAI's built-in
    OpenInference instrumentation to use TraceAI as the backend.

    Important: Call instrument() BEFORE importing any BeeAI modules.
    """

    _instance: Optional["BeeAIInstrumentorWrapper"] = None
    _is_instrumented: bool = False

    def __new__(cls) -> "BeeAIInstrumentorWrapper":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._tracer_provider: Optional[TracerProvider] = None
        self._beeai_instrumentor: Any = None

    def instrument(
        self,
        tracer_provider: Optional[TracerProvider] = None,
        otlp_endpoint: Optional[str] = None,
        otlp_headers: Optional[Dict[str, str]] = None,
    ) -> "BeeAIInstrumentorWrapper":
        """Configure BeeAI to send traces to TraceAI.

        This method sets up OpenTelemetry and the BeeAI instrumentor.
        Must be called BEFORE importing BeeAI modules.

        Args:
            tracer_provider: Optional OpenTelemetry tracer provider.
            otlp_endpoint: OTLP endpoint URL. Defaults to
                OTEL_EXPORTER_OTLP_TRACES_ENDPOINT or http://localhost:4318/v1/traces.
            otlp_headers: Headers for OTLP export (e.g., authentication).

        Returns:
            Self for method chaining.
        """
        if self._is_instrumented:
            return self

        # Set environment variables for OTLP export
        if otlp_endpoint:
            os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = otlp_endpoint

        if otlp_headers:
            header_str = ",".join(f"{k}={v}" for k, v in otlp_headers.items())
            os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = header_str

        # Set up tracer provider
        if tracer_provider:
            self._tracer_provider = tracer_provider
            trace.set_tracer_provider(tracer_provider)
        else:
            self._tracer_provider = trace.get_tracer_provider()

        # Configure OpenInference BeeAI instrumentor
        try:
            self._setup_beeai_instrumentor()
        except ImportError:
            # openinference-instrumentation-beeai not installed
            pass

        self._is_instrumented = True
        return self

    def _setup_beeai_instrumentor(self) -> None:
        """Set up the BeeAI OpenInference instrumentor."""
        try:
            from openinference.instrumentation.beeai import BeeAIInstrumentor

            self._beeai_instrumentor = BeeAIInstrumentor()
            self._beeai_instrumentor.instrument()
        except ImportError:
            raise ImportError(
                "openinference-instrumentation-beeai is not installed. "
                "Install it with: pip install openinference-instrumentation-beeai"
            )

    def uninstrument(self) -> None:
        """Remove instrumentation."""
        if not self._is_instrumented:
            return

        if self._beeai_instrumentor:
            try:
                self._beeai_instrumentor.uninstrument()
            except Exception:
                pass

        self._is_instrumented = False
        self._tracer_provider = None

    @property
    def is_instrumented(self) -> bool:
        """Check if instrumentation is active."""
        return self._is_instrumented


def configure_beeai_tracing(
    tracer_provider: Optional[TracerProvider] = None,
    otlp_endpoint: Optional[str] = None,
    otlp_headers: Optional[Dict[str, str]] = None,
    project_name: Optional[str] = None,
) -> BeeAIInstrumentorWrapper:
    """Configure BeeAI Framework to send traces to TraceAI.

    This is the main entry point for integrating TraceAI with BeeAI.
    Must be called BEFORE importing BeeAI modules.

    Example usage:

        from fi_instrumentation import register
        from fi_instrumentation.fi_types import ProjectType
        from traceai_beeai import configure_beeai_tracing

        # Option 1: Use with fi_instrumentation
        trace_provider = register(
            project_type=ProjectType.OBSERVE,
            project_name="beeai-agent",
        )
        configure_beeai_tracing(tracer_provider=trace_provider)

        # Option 2: Configure directly with endpoint
        configure_beeai_tracing(
            otlp_endpoint="https://api.traceai.com/v1/traces",
            otlp_headers={"Authorization": "Bearer YOUR_API_KEY"},
            project_name="beeai-agent",
        )

        # IMPORTANT: Now import BeeAI modules
        from beeai_framework.agents import Agent

    Args:
        tracer_provider: Optional OpenTelemetry tracer provider.
        otlp_endpoint: OTLP endpoint URL for trace export.
        otlp_headers: Headers for OTLP export (e.g., authentication).
        project_name: Optional project name for trace attributes.

    Returns:

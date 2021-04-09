"""Instrumentor for AWS Strands Agents.

This module provides functions to configure Strands' built-in OpenTelemetry
support to send traces to TraceAI, along with optional method wrapping for
additional instrumentation.
"""

import os
from typing import Any, Dict, List, Optional

from opentelemetry import trace
from opentelemetry.trace import TracerProvider

from traceai_strands._attributes import (
    SpanAttributes,
    get_agent_attributes,
    get_tool_attributes,
    get_model_attributes,
    get_trace_attributes_from_config,
)


class StrandsInstrumentor:
    """Instrumentor for AWS Strands Agents.

    This class provides methods to configure Strands' built-in telemetry
    to use TraceAI as the backend, and optionally wrap agent methods for
    additional observability.

    Strands has native OpenTelemetry support via the StrandsTelemetry class.
    This instrumentor:
    1. Configures environment variables for OTLP export
    2. Provides helper methods to set up StrandsTelemetry with TraceAI
    3. Optionally wraps Agent methods for extended capture
    """

    _instance: Optional["StrandsInstrumentor"] = None
    _is_instrumented: bool = False

    def __new__(cls) -> "StrandsInstrumentor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._tracer_provider: Optional[TracerProvider] = None
        self._original_methods: Dict[str, Any] = {}

    def instrument(
        self,
        tracer_provider: Optional[TracerProvider] = None,
        otlp_endpoint: Optional[str] = None,
        otlp_headers: Optional[Dict[str, str]] = None,
        enable_console_exporter: bool = False,
        enable_metrics: bool = True,
    ) -> "StrandsInstrumentor":
        """Configure Strands to send traces to TraceAI.

        This method sets up the necessary environment variables and
        optionally configures Strands' StrandsTelemetry class.

        Args:
            tracer_provider: Optional OpenTelemetry tracer provider.
            otlp_endpoint: OTLP endpoint URL. If not provided, uses
                OTEL_EXPORTER_OTLP_ENDPOINT environment variable.
            otlp_headers: Headers for OTLP export (e.g., authentication).
            enable_console_exporter: Whether to also export to console.
            enable_metrics: Whether to enable metrics collection.

        Returns:
            Self for method chaining.
        """
        if self._is_instrumented:
            return self

        self._tracer_provider = tracer_provider or trace.get_tracer_provider()

        # Set environment variables for Strands' native OTEL support
        if otlp_endpoint:
            os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = otlp_endpoint

        if otlp_headers:
            header_str = ",".join(f"{k}={v}" for k, v in otlp_headers.items())
            os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = header_str

        # Try to configure StrandsTelemetry if available
        try:
            self._configure_strands_telemetry(
                enable_console_exporter=enable_console_exporter,
                enable_metrics=enable_metrics,
            )
        except ImportError:
            # Strands not installed, environment variables are set
            pass

        self._is_instrumented = True
        return self

    def _configure_strands_telemetry(
        self,
        enable_console_exporter: bool = False,
        enable_metrics: bool = True,
    ) -> None:
        """Configure Strands' StrandsTelemetry class.

        Args:
            enable_console_exporter: Whether to enable console export.
            enable_metrics: Whether to enable metrics.
        """

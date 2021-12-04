"""TraceAI instrumentation for xAI (Grok).

This module provides OpenTelemetry instrumentation for xAI (Grok),
enabling comprehensive observability for xAI LLM API calls.
xAI uses an OpenAI-compatible API.

Example usage:

    from fi_instrumentation import register
    from fi_instrumentation.fi_types import ProjectType
    from traceai_xai import XAIInstrumentor

    # Setup TraceAI
    trace_provider = register(
        project_type=ProjectType.OBSERVE,
        project_name="xai-app",
    )

    # Instrument xAI
    XAIInstrumentor().instrument(tracer_provider=trace_provider)

    # Now use OpenAI client with xAI
    from openai import OpenAI

    client = OpenAI(
        api_key="your-xai-api-key",
        base_url="https://api.x.ai/v1",
    )
    response = client.chat.completions.create(
        model="grok-beta",
        messages=[{"role": "user", "content": "Hello!"}]
    )
"""

import logging
from importlib import import_module
from typing import Any, Collection

from fi_instrumentation import FITracer, TraceConfig
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from wrapt import wrap_function_wrapper

from traceai_xai.version import __version__
from traceai_xai._wrappers import (
    _CompletionsWrapper,
    _AsyncCompletionsWrapper,
)

"""TraceAI instrumentation for Fireworks AI.

This module provides OpenTelemetry instrumentation for Fireworks AI,
enabling comprehensive observability for Fireworks LLM API calls.
Fireworks uses an OpenAI-compatible API.

Example usage:

    from fi_instrumentation import register
    from fi_instrumentation.fi_types import ProjectType
    from traceai_fireworks import FireworksInstrumentor

    # Setup TraceAI
    trace_provider = register(
        project_type=ProjectType.OBSERVE,
        project_name="fireworks-app",
    )

    # Instrument Fireworks
    FireworksInstrumentor().instrument(tracer_provider=trace_provider)

    # Now use OpenAI client with Fireworks
    from openai import OpenAI

    client = OpenAI(
        api_key="your-fireworks-api-key",
        base_url="https://api.fireworks.ai/inference/v1",
    )
    response = client.chat.completions.create(
        model="accounts/fireworks/models/llama-v3p1-8b-instruct",
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

from traceai_fireworks.version import __version__
from traceai_fireworks._wrappers import (
    _CompletionsWrapper,
    _AsyncCompletionsWrapper,
)

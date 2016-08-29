"""Basic AutoGen v0.2 agent example with tracing.

This example demonstrates:
- Setting up instrumentation for AutoGen v0.2
- Creating AssistantAgent and UserProxyAgent
- Running a simple conversation with automatic tracing
"""

import os

# Setup tracing first
from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from traceai_autogen import AutogenInstrumentor

# Register with FutureAGI (or use any OTEL backend)
trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name="autogen-basic-example",
)

# Instrument AutoGen
AutogenInstrumentor().instrument(tracer_provider=trace_provider)

# Now import and use AutoGen
import autogen

# Configure LLM

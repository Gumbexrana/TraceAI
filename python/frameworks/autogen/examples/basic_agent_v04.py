"""Basic AutoGen v0.4 (AgentChat) example with tracing.

This example demonstrates:
- Setting up instrumentation for AutoGen v0.4
- Creating AssistantAgent with OpenAI model
- Running a simple agent query with automatic tracing
"""

import asyncio
import os

# Setup tracing first
from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from traceai_autogen import instrument_autogen

# Register and instrument
trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name="autogen-v04-basic",
)
instrument_autogen(tracer_provider=trace_provider)

# Now import AutoGen v0.4

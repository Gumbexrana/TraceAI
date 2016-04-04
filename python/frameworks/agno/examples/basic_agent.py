"""Basic Agno agent example with TraceAI instrumentation.

This example demonstrates how to set up tracing for a simple Agno agent.
"""

import os

# Setup tracing BEFORE importing Agno
from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from traceai_agno import configure_agno_tracing

# Initialize TraceAI
trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name="agno-basic-example",
)

# Configure Agno to use TraceAI
configure_agno_tracing(tracer_provider=trace_provider)

# Now import Agno modules

"""FutureAGI Platform integration example.

This example demonstrates how to use traceai-pydantic-ai with
the FutureAGI observability platform for production monitoring.
"""

import os
from pydantic import BaseModel, Field

# Check for FutureAGI API key
if not os.environ.get("FI_API_KEY"):
    print("Note: Set FI_API_KEY environment variable for FutureAGI integration")
    print("Running in demo mode with console output...\n")


class TaskResult(BaseModel):
    """Structured task result."""

    task_id: str = Field(description="Unique task identifier")
    status: str = Field(description="Task status: success, failed, pending")
    result: str = Field(description="Task result or error message")
    duration_seconds: float = Field(description="Task execution time")


def setup_futureagi_tracing():
    """Setup tracing with FutureAGI platform."""
    api_key = os.environ.get("FI_API_KEY")

    if api_key:
        try:
            from fi_instrumentation import register
            from fi_instrumentation.fi_types import ProjectType

            trace_provider = register(
                api_key=api_key,
                project_type=ProjectType.OBSERVE,
                project_name="pydantic-ai-demo",
            )
            print("Connected to FutureAGI platform")
            return trace_provider
        except ImportError:
            print("fi-instrumentation not installed, using console output")

    # Fallback to console exporter
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor


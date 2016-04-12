"""Agno workflow example with TraceAI instrumentation.

This example demonstrates how to set up tracing for complex workflows
with multiple steps and state management.
"""

import os

# Setup tracing BEFORE importing Agno
from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from traceai_agno import configure_agno_tracing, create_trace_context

# Initialize TraceAI
trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name="agno-workflow-example",
)

# Configure Agno to use TraceAI
configure_agno_tracing(tracer_provider=trace_provider)

# Now import Agno modules
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.workflow import Workflow


def main():
    """Run a workflow example with multiple steps."""
    # Create agents for different workflow steps
    analyzer = Agent(
        name="Analyzer",
        model=OpenAIChat(id="gpt-4"),
        description="Analyzes input and extracts key information.",
        instructions=["Analyze the input carefully", "Extract key points and themes"],
    )

    processor = Agent(
        name="Processor",
        model=OpenAIChat(id="gpt-4"),
        description="Processes analyzed data and generates insights.",
        instructions=["Use the analysis to generate insights", "Identify patterns"],

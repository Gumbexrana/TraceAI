"""Agno multi-agent team example with TraceAI instrumentation.

This example demonstrates how to set up tracing for a team of agents
working together to accomplish complex tasks.
"""

import os

# Setup tracing BEFORE importing Agno
from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from traceai_agno import configure_agno_tracing, get_team_attributes

# Initialize TraceAI
trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name="agno-team-example",
)

# Configure Agno to use TraceAI
configure_agno_tracing(tracer_provider=trace_provider)

# Now import Agno modules
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.team import Team


def main():
    """Run a multi-agent team example."""
    # Create specialized agents
    researcher = Agent(
        name="Researcher",
        model=OpenAIChat(id="gpt-4"),
        description="A research specialist that gathers and analyzes information.",
        instructions=[

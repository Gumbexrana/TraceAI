"""AutoGen v0.4 RoundRobinGroupChat example with tracing.

This example demonstrates:
- Creating multiple specialized agents
- Orchestrating them with RoundRobinGroupChat
- Tracing team interactions and message flow
"""

import asyncio
import os

from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from traceai_autogen import instrument_autogen

# Setup tracing
trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name="autogen-team-roundrobin",
)
instrument_autogen(tracer_provider=trace_provider)

# Import AutoGen v0.4
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def main():
    # Create model client
    model = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Create specialized agents for a code review team
    coder = AssistantAgent(
        name="coder",

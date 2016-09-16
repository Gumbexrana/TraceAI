"""AutoGen v0.4 SelectorGroupChat example with tracing.

This example demonstrates:
- Creating a team with dynamic agent selection
- Using SelectorGroupChat for intelligent routing
- Tracing selector decisions and agent interactions
"""

import asyncio
import os

from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from traceai_autogen import instrument_autogen

# Setup tracing
trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name="autogen-team-selector",
)
instrument_autogen(tracer_provider=trace_provider)

# Import AutoGen v0.4
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def main():
    # Create model client
    model = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Create specialized agents for a software development team
    architect = AssistantAgent(
        name="architect",
        description="Software architect who designs system architecture and high-level structure.",
        model_client=model,
        system_message="""You are a software architect.
        Focus on:
        - System design and architecture
        - Technology stack decisions
        - Scalability considerations
        - Design patterns
        Always start by understanding requirements before proposing solutions.""",
    )

    backend_dev = AssistantAgent(
        name="backend_developer",
        description="Backend developer who implements server-side logic and APIs.",

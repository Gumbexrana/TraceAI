from __future__ import annotations

import json
import random

from agents import Agent, HandoffInputData, Runner, function_tool, handoff, trace
from agents.extensions import handoff_filters
from fi_instrumentation import register
from traceai_openai_agents import OpenAIAgentsInstrumentor

tracer_provider = register(project_name="openai-agents-example")
OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)


@function_tool
def random_number_tool(max: int) -> int:
    """Return a random integer between 0 and the given maximum."""
    return random.randint(0, max)


def spanish_handoff_message_filter(
    handoff_message_data: HandoffInputData,
) -> HandoffInputData:
    # First, we'll remove any tool-related messages from the message history
    handoff_message_data = handoff_filters.remove_all_tools(handoff_message_data)

    # Second, we'll also remove the first two items from the history, just for demonstration
    history = (
        tuple(handoff_message_data.input_history[2:])
        if isinstance(handoff_message_data.input_history, tuple)
        else handoff_message_data.input_history
    )

    return HandoffInputData(
        input_history=history,
        pre_handoff_items=tuple(handoff_message_data.pre_handoff_items),
        new_items=tuple(handoff_message_data.new_items),
    )


first_agent = Agent(
    name="Assistant",
    instructions="Be extremely concise.",
    tools=[random_number_tool],
)

spanish_agent = Agent(
    name="Spanish Assistant",
    instructions="You only speak Spanish and are extremely concise.",
    handoff_description="A Spanish-speaking assistant.",
)

second_agent = Agent(
    name="Assistant",
    instructions=(
        "Be a helpful assistant. If the user speaks Spanish, handoff to the Spanish assistant."
    ),
    handoffs=[handoff(spanish_agent, input_filter=spanish_handoff_message_filter)],
)


async def main():
    # Trace the entire run as a single workflow
    with trace(workflow_name="Streaming message filter"):
        # 1. Send a regular message to the first agent
        result = await Runner.run(first_agent, input="Hi, my name is Sora.")

        print("Step 1 done")

        # 2. Ask it to square a number
        result = await Runner.run(
            second_agent,
            input=result.to_input_list()
            + [
                {

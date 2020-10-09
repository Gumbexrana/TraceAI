"""Conversation with memory example.

This example demonstrates maintaining conversation history
with OpenTelemetry tracing for each turn.
"""

import asyncio
from typing import List, Dict

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from traceai_pydantic_ai import PydanticAIInstrumentor


def setup_tracing():
    """Setup OpenTelemetry with console exporter for demo."""
    provider = TracerProvider()
    processor = SimpleSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    return provider


class ConversationManager:
    """Manage conversation history and agent interactions."""

    def __init__(self, agent):
        self.agent = agent
        self.history: List[Dict] = []

    async def send_message(self, user_message: str) -> str:
        """Send a message and get response, maintaining history."""
        # Run agent with conversation history
        result = await self.agent.run(
            user_message,

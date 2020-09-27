"""Pydantic AI agent with tools example.

This example demonstrates tool usage with OpenTelemetry tracing.
Tools are automatically traced when instrumentation is enabled.
"""

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


def main():
    # Setup tracing
    provider = setup_tracing()

    # Initialize Pydantic AI instrumentation
    PydanticAIInstrumentor().instrument(tracer_provider=provider)

    # Import after instrumentation
    from pydantic_ai import Agent, RunContext

    # Create agent
    agent = Agent(
        'openai:gpt-4o-mini',

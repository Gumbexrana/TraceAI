"""Pydantic AI multi-model example.

This example demonstrates using different model providers with tracing.
Model provider is automatically detected from model names.
"""

import asyncio

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from traceai_pydantic_ai import PydanticAIInstrumentor, get_model_provider


def setup_tracing():
    """Setup OpenTelemetry with console exporter for demo."""
    provider = TracerProvider()
    processor = SimpleSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    return provider


async def query_agent(agent, prompt: str, model_name: str):
    """Query an agent and print results."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Provider: {get_model_provider(model_name)}")
    print(f"Prompt: {prompt}")

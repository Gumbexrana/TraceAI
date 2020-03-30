from typing import Iterator

import pytest
from traceai_openai_agents import OpenAIAgentsInstrumentor
from traceai_openai import OpenAIInstrumentor

from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture
def in_memory_span_exporter() -> InMemorySpanExporter:

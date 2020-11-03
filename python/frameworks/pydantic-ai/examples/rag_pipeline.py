"""RAG Pipeline example with Pydantic AI and tracing.

This example demonstrates a Retrieval Augmented Generation (RAG) pipeline
with full OpenTelemetry tracing for each step.
"""

from typing import List
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from pydantic import BaseModel, Field

from traceai_pydantic_ai import PydanticAIInstrumentor


class Document(BaseModel):
    """A retrieved document."""

    title: str
    content: str
    relevance_score: float


class RAGResponse(BaseModel):
    """Structured response from RAG pipeline."""

    answer: str = Field(description="The answer to the user's question")
    sources: List[str] = Field(description="List of source documents used")
    confidence: float = Field(description="Confidence score 0-1", ge=0, le=1)


def setup_tracing():
    """Setup OpenTelemetry with console exporter for demo."""
    provider = TracerProvider()
    processor = SimpleSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    return provider


# Simulated document database
KNOWLEDGE_BASE = [
    Document(
        title="Python Best Practices",
        content="Use type hints for better code quality. Follow PEP 8 style guide.",
        relevance_score=0.95,
    ),

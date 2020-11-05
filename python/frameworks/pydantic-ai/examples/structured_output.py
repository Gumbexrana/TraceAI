"""Pydantic AI structured output example.

This example demonstrates using structured outputs with Pydantic models.
The result type is traced as part of the span attributes.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from pydantic import BaseModel, Field

from traceai_pydantic_ai import PydanticAIInstrumentor


class MovieRecommendation(BaseModel):
    """A movie recommendation."""

    title: str = Field(description="The movie title")
    year: int = Field(description="Release year")
    genre: str = Field(description="Primary genre")
    director: str = Field(description="Director name")
    rating: float = Field(description="Rating out of 10", ge=0, le=10)
    summary: str = Field(description="Brief plot summary")
    why_recommended: str = Field(description="Why this movie is recommended")


def setup_tracing():
    """Setup OpenTelemetry with console exporter for demo."""
    provider = TracerProvider()
    processor = SimpleSpanProcessor(ConsoleSpanExporter())

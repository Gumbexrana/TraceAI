"""Together AI embeddings example with tracing."""
import os
from together import Together
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from traceai_together import TogetherInstrumentor

# Set up tracing
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# Instrument Together
TogetherInstrumentor().instrument(tracer_provider=provider)


def cosine_similarity(a: list, b: list) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0


def main():
    client = Together()

    # Single embedding
    print("=== Single Embedding ===")
    response = client.embeddings.create(
        model="togethercomputer/m2-bert-80M-8k-retrieval",
        input="Hello, world!",
    )
    embedding = response.data[0].embedding
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")

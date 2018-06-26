"""HuggingFace feature extraction (embeddings) example with tracing."""
import os
from huggingface_hub import InferenceClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from traceai_huggingface import HuggingFaceInstrumentor

# Set up tracing
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# Instrument HuggingFace
HuggingFaceInstrumentor().instrument(tracer_provider=provider)


def cosine_similarity(a: list, b: list) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0


def main():
    client = InferenceClient()

    # Single embedding
    print("=== Single Embedding ===")
    embedding = client.feature_extraction(
        text="Hello, world!",
        model="sentence-transformers/all-MiniLM-L6-v2",
    )
    # Convert to list if needed (may be numpy array)
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()

    # Handle nested structure
    if isinstance(embedding[0], list):
        embedding = embedding[0]

    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")

    # Semantic similarity

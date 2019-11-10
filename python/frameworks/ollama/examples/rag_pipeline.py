"""RAG pipeline example with Ollama and tracing."""
import ollama
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from traceai_ollama import OllamaInstrumentor

# Set up tracing
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# Instrument Ollama
OllamaInstrumentor().instrument(tracer_provider=provider)


def cosine_similarity(a: list, b: list) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0


def main():
    # Knowledge base
    documents = [
        "Python is a high-level programming language known for its readability.",
        "Machine learning is a subset of artificial intelligence.",
        "The Eiffel Tower is located in Paris, France.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning uses neural networks with many layers.",
    ]

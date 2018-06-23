"""Basic HuggingFace Inference API example with tracing."""
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


def main():
    # Initialize client (uses HF_TOKEN environment variable)
    client = InferenceClient()

    # Text Generation
    print("=== Text Generation ===")
    response = client.text_generation(
        "The capital of France is",
        model="meta-llama/Llama-2-7b-chat-hf",

"""HuggingFace chat completions example with tracing."""
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
    client = InferenceClient()

    # Simple chat
    print("=== Simple Chat ===")
    response = client.chat_completion(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_tokens=100,
    )
    print(f"Response: {response.choices[0].message.content}")
    print(f"Tokens - Prompt: {response.usage.prompt_tokens}")
    print(f"Tokens - Completion: {response.usage.completion_tokens}")

    # Multi-turn conversation

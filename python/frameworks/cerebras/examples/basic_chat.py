"""Basic Cerebras chat completion example with TraceAI instrumentation."""

import os

from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from traceai_cerebras import CerebrasInstrumentor

# Setup TraceAI
trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name="cerebras-basic-example",
)

# Instrument Cerebras
CerebrasInstrumentor().instrument(tracer_provider=trace_provider)

# Now import and use Cerebras
from cerebras.cloud.sdk import Cerebras


def main():
    """Run a basic chat completion example."""
    client = Cerebras()

    print("Cerebras Chat Completion Example")
    print("-" * 40)

    # Simple chat completion
    response = client.chat.completions.create(
        model="llama3.1-8b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_tokens=100,
        temperature=0.7,
    )

    print(f"Model: {response.model}")
    print(f"Response: {response.choices[0].message.content}")
    print()

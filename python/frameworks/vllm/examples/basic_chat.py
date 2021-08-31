"""Basic chat example with vLLM instrumentation."""

import os

from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from traceai_vllm import VLLMInstrumentor


def main():
    # Register TraceAI
    trace_provider = register(
        project_type=ProjectType.OBSERVE,
        project_name="vllm-example",
    )

    # Instrument vLLM (specify your vLLM server URL)
    VLLMInstrumentor(
        vllm_base_urls=["localhost:8000"]  # Your vLLM server(s)
    ).instrument(tracer_provider=trace_provider)

    # Now use OpenAI client with vLLM
    from openai import OpenAI

    client = OpenAI(
        api_key="token",  # vLLM doesn't require a real API key
        base_url="http://localhost:8000/v1",
    )

    # Make a chat completion request
    response = client.chat.completions.create(
        model="meta-llama/Llama-2-7b-chat-hf",  # Model loaded in vLLM

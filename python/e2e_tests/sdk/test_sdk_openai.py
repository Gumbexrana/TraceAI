"""
E2E Tests for OpenAI SDK Instrumentation

Tests OpenAI API calls and verifies traces are correctly captured.
"""

import pytest
import os
import time
import asyncio
from typing import Dict, Any

from config import config, skip_if_no_openai


@pytest.fixture(scope="module")
def openai_client():
    """Create instrumented OpenAI client."""
    if not config.has_openai():
        pytest.skip("OpenAI API key not available")

    os.environ["OPENAI_API_KEY"] = config.openai_api_key

    # Import and instrument
    from fi_instrumentation import register
    try:
        from traceai_openai import OpenAIInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_openai not installed or incompatible")

    # Register tracer - sends to FutureAGI cloud
    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,  # Reduce output noise
    )

    # Instrument OpenAI
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

    # Create client
    from openai import OpenAI

    client = OpenAI()

    yield client

    # Cleanup
    OpenAIInstrumentor().uninstrument()


@pytest.fixture(scope="module")
def async_openai_client():
    """Create instrumented async OpenAI client."""
    if not config.has_openai():
        pytest.skip("OpenAI API key not available")

    os.environ["OPENAI_API_KEY"] = config.openai_api_key

    from openai import AsyncOpenAI

    return AsyncOpenAI()


@skip_if_no_openai
class TestOpenAIChatCompletion:
    """Test OpenAI chat completion instrumentation."""

    def test_basic_chat_completion(self, openai_client):
        """Test basic chat completion creates correct spans."""
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Say 'Hello E2E Test' in exactly 3 words."}
            ],
            max_tokens=50,
        )

        # Verify response
        assert response.choices[0].message.content is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

        # Allow time for span export
        time.sleep(2)

        # Verify the response contains expected data
        print(f"Response: {response.choices[0].message.content}")
        print(f"Tokens - Prompt: {response.usage.prompt_tokens}, Completion: {response.usage.completion_tokens}")

    def test_chat_completion_with_system_message(self, openai_client):
        """Test chat with system message."""
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that responds briefly."},
                {"role": "user", "content": "What is 2+2?"},
            ],
            max_tokens=20,
        )

        assert response.choices[0].message.content is not None
        assert "4" in response.choices[0].message.content

    def test_chat_completion_streaming(self, openai_client):
        """Test streaming chat completion."""
        stream = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Count from 1 to 5."}
            ],
            max_tokens=50,
            stream=True,
        )

        chunks = []
        for chunk in stream:
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)

        full_response = "".join(chunks)
        assert len(full_response) > 0
        print(f"Streamed response: {full_response}")

    def test_chat_completion_with_function_calling(self, openai_client):
        """Test function/tool calling."""
        tools = [
            {
                "type": "function",

"""
E2E Tests for Anthropic SDK Instrumentation

Tests Anthropic API calls and verifies traces are correctly captured.
"""

import pytest
import os
import time
import asyncio
from typing import Dict, Any

from config import config, skip_if_no_anthropic


@pytest.fixture(scope="module")
def anthropic_client():
    """Create instrumented Anthropic client."""
    if not config.has_anthropic():
        pytest.skip("Anthropic API key not available")

    os.environ["ANTHROPIC_API_KEY"] = config.anthropic_api_key

    # Import and instrument
    from fi_instrumentation import register
    try:
        from traceai_anthropic import AnthropicInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_anthropic not installed or incompatible")

    # Register tracer - sends to FutureAGI cloud
    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    # Instrument Anthropic
    AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)

    # Create client
    from anthropic import Anthropic

    client = Anthropic()

    yield client

    # Cleanup
    AnthropicInstrumentor().uninstrument()


@pytest.fixture(scope="module")
def async_anthropic_client():
    """Create async Anthropic client."""
    if not config.has_anthropic():
        pytest.skip("Anthropic API key not available")

    os.environ["ANTHROPIC_API_KEY"] = config.anthropic_api_key

    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        pytest.skip("anthropic not installed")

    return AsyncAnthropic()


@skip_if_no_anthropic
class TestAnthropicMessages:
    """Test Anthropic messages API instrumentation."""

    def test_basic_message(self, anthropic_client):
        """Test basic message creation."""
        response = anthropic_client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=50,
            messages=[
                {"role": "user", "content": "Say 'Hello E2E Test' in exactly 3 words."}
            ],
        )

        # Verify response
        assert len(response.content) > 0
        assert response.content[0].type == "text"
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

        # Allow time for span export
        time.sleep(2)

        print(f"Response: {response.content[0].text}")
        print(f"Tokens - Input: {response.usage.input_tokens}, Output: {response.usage.output_tokens}")

    def test_message_with_system(self, anthropic_client):
        """Test message with system prompt."""
        response = anthropic_client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=30,
            system="You are a helpful assistant that responds in exactly one word.",
            messages=[
                {"role": "user", "content": "What is 2+2?"}
            ],
        )

        assert len(response.content) > 0
        text = response.content[0].text.strip()
        # Should be a brief response
        assert len(text.split()) <= 3

    def test_streaming_message(self, anthropic_client):
        """Test streaming messages."""
        chunks = []

        with anthropic_client.messages.stream(
            model="claude-3-5-haiku-latest",
            max_tokens=50,
            messages=[
                {"role": "user", "content": "Count from 1 to 5."}
            ],
        ) as stream:
            for text in stream.text_stream:
                chunks.append(text)

        full_response = "".join(chunks)
        assert len(full_response) > 0
        print(f"Streamed response: {full_response}")

    def test_multi_turn_conversation(self, anthropic_client):
        """Test multi-turn conversation."""
        messages = [
            {"role": "user", "content": "My name is Alice."},
        ]

        response1 = anthropic_client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=50,
            messages=messages,
        )

        # Add assistant response and follow-up
        messages.append({"role": "assistant", "content": response1.content[0].text})

"""
E2E Tests for Groq SDK Instrumentation

Tests Groq API calls and verifies traces are correctly captured.
"""

import pytest
import os
import time
from typing import Dict, Any

from config import config, skip_if_no_groq


@pytest.fixture(scope="module")
def groq_client():
    """Create instrumented Groq client."""
    if not config.has_groq():
        pytest.skip("Groq API key not available")

    os.environ["GROQ_API_KEY"] = config.groq_api_key

    # Import and instrument
    from fi_instrumentation import register
    try:
        from traceai_groq import GroqInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_groq not installed or incompatible")

    # Register tracer
    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    # Instrument Groq
    GroqInstrumentor().instrument(tracer_provider=tracer_provider)

    # Create client
    from groq import Groq

    client = Groq()

    yield client

    # Cleanup
    GroqInstrumentor().uninstrument()


@skip_if_no_groq
class TestGroqChatCompletion:
    """Test Groq chat completion instrumentation."""

    def test_basic_chat_completion(self, groq_client):
        """Test basic chat completion."""
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
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

        print(f"Response: {response.choices[0].message.content}")
        print(f"Tokens - Prompt: {response.usage.prompt_tokens}, Completion: {response.usage.completion_tokens}")

    def test_streaming_chat(self, groq_client):
        """Test streaming chat completion."""
        stream = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": "Count from 1 to 5."}
            ],
            max_tokens=50,
            stream=True,
        )

        chunks = []

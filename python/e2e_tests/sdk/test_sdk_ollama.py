"""
E2E Tests for Ollama SDK Instrumentation

Tests Ollama native SDK with tracing. Requires a local Ollama server.
Set OLLAMA_HOST to enable these tests.
"""

import pytest
import time

from config import config, skip_if_no_ollama


@pytest.fixture(scope="module")
def ollama_client():
    """Create an instrumented Ollama client."""
    if not config.has_ollama():
        pytest.skip("OLLAMA_HOST not set")

    from fi_instrumentation import register
    try:
        from traceai_ollama import OllamaInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_ollama not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    OllamaInstrumentor().instrument(tracer_provider=tracer_provider)

    import ollama

    client = ollama.Client(host=config.ollama_host)

    yield client

    OllamaInstrumentor().uninstrument()


@skip_if_no_ollama
class TestOllamaChatCompletion:
    """Test Ollama chat completion instrumentation."""

    def test_basic_chat(self, ollama_client):
        """Test basic chat completion."""
        response = ollama_client.chat(
            model="llama3.2",
            messages=[

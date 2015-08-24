"""
E2E Tests for Google GenAI SDK Instrumentation

Tests Google GenAI native SDK (google.genai.Client) with tracing.
Uses GOOGLE_API_KEY directly — no OpenAI-compat wrapper needed.
"""

import pytest
import time

from config import config, skip_if_no_google


@pytest.fixture(scope="module")
def genai_client():
    """Create an instrumented Google GenAI client."""
    if not config.has_google():
        pytest.skip("GOOGLE_API_KEY not set")

    from fi_instrumentation import register
    try:
        from traceai_google_genai import GoogleGenAIInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_google_genai not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)

    from google import genai

    client = genai.Client(api_key=config.google_api_key)

    yield client

    GoogleGenAIInstrumentor().uninstrument()


@pytest.fixture(scope="module")
def async_genai_client():
    """Return the same client — google.genai supports async via aio namespace."""
    if not config.has_google():
        pytest.skip("GOOGLE_API_KEY not set")

    from google import genai

    return genai.Client(api_key=config.google_api_key)


@skip_if_no_google
class TestGoogleGenAIGenerateContent:
    """Test Google GenAI generate_content instrumentation."""

    def test_basic_generate(self, genai_client):
        """Test basic content generation."""
        response = genai_client.models.generate_content(
            model=config.google_model,
            contents="Say 'Hello E2E Test' in exactly 3 words.",
        )

        assert response.text is not None
        assert len(response.text) > 0

        time.sleep(2)
        print(f"Response: {response.text}")

    def test_with_system_instruction(self, genai_client):
        """Test generation with system instruction."""
        from google.genai import types

        response = genai_client.models.generate_content(
            model=config.google_model,
            contents="What is 2+2?",
            config=types.GenerateContentConfig(
                system_instruction="You are a helpful assistant that responds briefly.",
                max_output_tokens=50,
            ),
        )

        assert response.text is not None
        assert "4" in response.text

    def test_multi_turn_conversation(self, genai_client):
        """Test multi-turn conversation with content list."""
        from google.genai import types

        contents = [
            types.Content(

"""
E2E Tests for Fireworks SDK Instrumentation

Tests Fireworks instrumentation using Google's OpenAI-compatible endpoint.
The Fireworks instrumentor wraps the openai.OpenAI SDK methods.
"""

import pytest
import time

from config import config, skip_if_no_google


@pytest.fixture(scope="module")
def fireworks_client():
    """Create an instrumented Fireworks (OpenAI-compat) client via Google endpoint."""
    if not config.has_google():
        pytest.skip("GOOGLE_API_KEY not set")

    from fi_instrumentation import register
    try:
        from traceai_fireworks import FireworksInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_fireworks not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    FireworksInstrumentor().instrument(tracer_provider=tracer_provider)

    from openai import OpenAI

    client = OpenAI(
        base_url=config.google_openai_base_url,
        api_key=config.google_api_key,
    )

    yield client

    FireworksInstrumentor().uninstrument()


@pytest.fixture(scope="module")
def async_fireworks_client():
    """Create an async Fireworks (OpenAI-compat) client via Google endpoint."""
    if not config.has_google():
        pytest.skip("GOOGLE_API_KEY not set")

    from openai import AsyncOpenAI

    return AsyncOpenAI(
        base_url=config.google_openai_base_url,
        api_key=config.google_api_key,
    )


@skip_if_no_google
class TestFireworksChatCompletion:
    """Test Fireworks chat completion instrumentation."""

    def test_basic_chat(self, fireworks_client):
        """Test basic chat completion."""
        response = fireworks_client.chat.completions.create(
            model=config.google_model,
            messages=[
                {"role": "user", "content": "Say 'Hello E2E Test' in exactly 3 words."}
            ],
            max_tokens=50,
        )

        assert response.choices[0].message.content is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0


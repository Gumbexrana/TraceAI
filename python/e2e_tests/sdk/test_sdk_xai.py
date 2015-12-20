"""
E2E Tests for XAI SDK Instrumentation

Tests XAI instrumentation using Google's OpenAI-compatible endpoint.
The XAI instrumentor wraps the openai.OpenAI SDK methods, so pointing
base_url to Google's endpoint exercises the instrumentor without an XAI key.
"""

import pytest
import time

from config import config, skip_if_no_google


@pytest.fixture(scope="module")
def xai_client():
    """Create an instrumented XAI (OpenAI-compat) client via Google endpoint."""
    if not config.has_google():
        pytest.skip("GOOGLE_API_KEY not set")

    from fi_instrumentation import register
    try:
        from traceai_xai import XAIInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_xai not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    XAIInstrumentor().instrument(tracer_provider=tracer_provider)

    from openai import OpenAI

    client = OpenAI(
        base_url=config.google_openai_base_url,
        api_key=config.google_api_key,
    )

    yield client

    XAIInstrumentor().uninstrument()


@pytest.fixture(scope="module")
def async_xai_client():
    """Create an async XAI (OpenAI-compat) client via Google endpoint."""
    if not config.has_google():
        pytest.skip("GOOGLE_API_KEY not set")

    from openai import AsyncOpenAI

    return AsyncOpenAI(
        base_url=config.google_openai_base_url,
        api_key=config.google_api_key,
    )


@skip_if_no_google
class TestXAIChatCompletion:
    """Test XAI chat completion instrumentation."""

    def test_basic_chat(self, xai_client):
        """Test basic chat completion."""
        response = xai_client.chat.completions.create(
            model=config.google_model,
            messages=[
                {"role": "user", "content": "Say 'Hello E2E Test' in exactly 3 words."}
            ],
            max_tokens=50,
        )

        assert response.choices[0].message.content is not None

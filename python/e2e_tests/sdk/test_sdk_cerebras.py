"""
E2E Tests for Cerebras SDK Instrumentation

Tests Cerebras instrumentation using Google's OpenAI-compatible endpoint.
The Cerebras instrumentor wraps the openai.OpenAI SDK methods.
"""

import pytest
import time

from config import config, skip_if_no_google


@pytest.fixture(scope="module")
def cerebras_client():
    """Create an instrumented Cerebras (OpenAI-compat) client via Google endpoint."""
    if not config.has_google():
        pytest.skip("GOOGLE_API_KEY not set")

    from fi_instrumentation import register
    try:
        from traceai_cerebras import CerebrasInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_cerebras not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    CerebrasInstrumentor().instrument(tracer_provider=tracer_provider)

    from openai import OpenAI

    client = OpenAI(
        base_url=config.google_openai_base_url,
        api_key=config.google_api_key,
    )

    yield client

    CerebrasInstrumentor().uninstrument()


@pytest.fixture(scope="module")
def async_cerebras_client():
    """Create an async Cerebras (OpenAI-compat) client via Google endpoint."""
    if not config.has_google():
        pytest.skip("GOOGLE_API_KEY not set")

    from openai import AsyncOpenAI

    return AsyncOpenAI(
        base_url=config.google_openai_base_url,
        api_key=config.google_api_key,
    )


@skip_if_no_google
class TestCerebrasChatCompletion:
    """Test Cerebras chat completion instrumentation."""

    def test_basic_chat(self, cerebras_client):
        """Test basic chat completion."""
        response = cerebras_client.chat.completions.create(
            model=config.google_model,
            messages=[
                {"role": "user", "content": "Say 'Hello E2E Test' in exactly 3 words."}
            ],
            max_tokens=50,
        )

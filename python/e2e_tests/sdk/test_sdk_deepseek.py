"""
E2E Tests for DeepSeek SDK Instrumentation

Tests DeepSeek instrumentation using Google's OpenAI-compatible endpoint.
The DeepSeek instrumentor wraps the openai.OpenAI SDK methods.
"""

import pytest
import time

from config import config, skip_if_no_google


@pytest.fixture(scope="module")
def deepseek_client():
    """Create an instrumented DeepSeek (OpenAI-compat) client via Google endpoint."""
    if not config.has_google():
        pytest.skip("GOOGLE_API_KEY not set")

    from fi_instrumentation import register
    try:
        from traceai_deepseek import DeepSeekInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_deepseek not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    DeepSeekInstrumentor().instrument(tracer_provider=tracer_provider)

    from openai import OpenAI

    client = OpenAI(
        base_url=config.google_openai_base_url,
        api_key=config.google_api_key,
    )

    yield client

    DeepSeekInstrumentor().uninstrument()


@pytest.fixture(scope="module")
def async_deepseek_client():
    """Create an async DeepSeek (OpenAI-compat) client via Google endpoint."""
    if not config.has_google():
        pytest.skip("GOOGLE_API_KEY not set")

    from openai import AsyncOpenAI

    return AsyncOpenAI(
        base_url=config.google_openai_base_url,
        api_key=config.google_api_key,
    )


@skip_if_no_google
class TestDeepSeekChatCompletion:
    """Test DeepSeek chat completion instrumentation."""

    def test_basic_chat(self, deepseek_client):
        """Test basic chat completion."""
        response = deepseek_client.chat.completions.create(
            model=config.google_model,
            messages=[
                {"role": "user", "content": "Say 'Hello E2E Test' in exactly 3 words."}
            ],
            max_tokens=50,
        )

        assert response.choices[0].message.content is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0


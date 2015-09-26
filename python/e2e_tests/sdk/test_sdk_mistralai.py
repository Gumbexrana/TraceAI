"""
E2E Tests for MistralAI SDK Instrumentation

Tests MistralAI native SDK with tracing. Requires MISTRAL_API_KEY.
"""

import pytest
import time

from config import config, skip_if_no_mistral


@pytest.fixture(scope="module")
def mistral_client():
    """Create an instrumented Mistral client."""
    if not config.has_mistral():
        pytest.skip("MISTRAL_API_KEY not set")

    from fi_instrumentation import register
    try:
        from traceai_mistralai import MistralAIInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_mistralai not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    MistralAIInstrumentor().instrument(tracer_provider=tracer_provider)

    from mistralai import Mistral

    client = Mistral(api_key=config.mistral_api_key)

    yield client

    MistralAIInstrumentor().uninstrument()


@skip_if_no_mistral
class TestMistralChatCompletion:
    """Test Mistral chat completion instrumentation."""

    def test_basic_chat(self, mistral_client):
        """Test basic chat completion."""
        response = mistral_client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "user", "content": "Say 'Hello E2E Test' in exactly 3 words."}
            ],
            max_tokens=50,
        )


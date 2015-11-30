"""
E2E Tests for Together SDK Instrumentation

Tests Together native SDK with tracing. Requires TOGETHER_API_KEY.
"""

import pytest
import time

from config import config, skip_if_no_together


@pytest.fixture(scope="module")
def together_client():
    """Create an instrumented Together client."""
    if not config.has_together():
        pytest.skip("TOGETHER_API_KEY not set")

    from fi_instrumentation import register
    try:
        from traceai_together import TogetherInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_together not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    TogetherInstrumentor().instrument(tracer_provider=tracer_provider)

    import together

    client = together.Together(api_key=config.together_api_key)

    yield client

    TogetherInstrumentor().uninstrument()


@skip_if_no_together
class TestTogetherChatCompletion:
    """Test Together chat completion instrumentation."""

    def test_basic_chat(self, together_client):
        """Test basic chat completion."""
        response = together_client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[
                {"role": "user", "content": "Say 'Hello E2E Test' in exactly 3 words."}
            ],
            max_tokens=50,
        )

        assert response.choices[0].message.content is not None

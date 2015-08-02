"""
E2E Tests for Cohere SDK Instrumentation

Tests Cohere native SDK with tracing. Requires COHERE_API_KEY.
"""

import pytest
import time

from config import config, skip_if_no_cohere


@pytest.fixture(scope="module")
def cohere_client():
    """Create an instrumented Cohere client."""
    if not config.has_cohere():
        pytest.skip("COHERE_API_KEY not set")

    from fi_instrumentation import register
    try:
        from traceai_cohere import CohereInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_cohere not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    CohereInstrumentor().instrument(tracer_provider=tracer_provider)

    import cohere

    client = cohere.ClientV2(api_key=config.cohere_api_key)

    yield client

    CohereInstrumentor().uninstrument()


@skip_if_no_cohere
class TestCohereChatCompletion:
    """Test Cohere chat completion instrumentation."""

    def test_basic_chat(self, cohere_client):
        """Test basic chat."""
        response = cohere_client.chat(
            model="command-r-plus",
            messages=[
                {"role": "user", "content": "Say 'Hello E2E Test' in exactly 3 words."}

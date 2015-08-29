"""
E2E Tests for Haystack SDK Instrumentation

Tests Haystack instrumentation using Google's OpenAI-compatible endpoint.
"""

import pytest
import time

from config import config, skip_if_no_google


@pytest.fixture(scope="module")
def setup_haystack():
    """Set up Haystack with instrumentation."""
    if not config.has_google():
        pytest.skip("GOOGLE_API_KEY not set")

    from fi_instrumentation import register
    try:
        from traceai_haystack import HaystackInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_haystack not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    HaystackInstrumentor().instrument(tracer_provider=tracer_provider)

    yield

    HaystackInstrumentor().uninstrument()


@skip_if_no_google
class TestHaystackGenerator:
    """Test Haystack OpenAI generator."""

    def test_basic_generation(self, setup_haystack):
        """Test basic text generation."""
        from haystack.components.generators import OpenAIGenerator

        generator = OpenAIGenerator(
            model=config.google_model,
            api_base_url=config.google_openai_base_url,
            api_key=config.google_api_key,
            generation_kwargs={"max_tokens": 50},
        )

        result = generator.run(prompt="Say 'Hello E2E Test' in exactly 3 words.")

        assert "replies" in result
        assert len(result["replies"]) > 0
        assert result["replies"][0] is not None
        time.sleep(2)
        print(f"Response: {result['replies'][0]}")

    def test_chat_generation(self, setup_haystack):
        """Test chat generation."""
        from haystack.components.generators.chat import OpenAIChatGenerator

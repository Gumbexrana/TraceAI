"""
E2E Tests for LlamaIndex SDK Instrumentation

Tests LlamaIndex instrumentation using Google's OpenAI-compatible endpoint.
"""

import pytest
import time

from config import config, skip_if_no_google


@pytest.fixture(scope="module")
def setup_llamaindex():
    """Set up LlamaIndex with instrumentation."""
    if not config.has_google():
        pytest.skip("GOOGLE_API_KEY not set")

    from fi_instrumentation import register
    try:
        from traceai_llama_index import LlamaIndexInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_llama_index not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

    yield

    LlamaIndexInstrumentor().uninstrument()


@skip_if_no_google
class TestLlamaIndexLLM:
    """Test LlamaIndex LLM integration."""

    def test_basic_llm_complete(self, setup_llamaindex):
        """Test basic LLM completion."""
        from llama_index.llms.openai import OpenAI

        llm = OpenAI(
            model=config.google_model,
            api_base=config.google_openai_base_url,
            api_key=config.google_api_key,
            max_tokens=50,
        )

        response = llm.complete("Say 'Hello E2E Test' in exactly 3 words.")

        assert response.text is not None
        assert len(response.text) > 0
        time.sleep(2)
        print(f"Response: {response.text}")

    def test_llm_chat(self, setup_llamaindex):
        """Test LLM chat interface."""

"""
E2E Tests for PydanticAI SDK Instrumentation

Tests PydanticAI instrumentation using Google's OpenAI-compatible endpoint.
"""

import pytest
import time

from config import config, skip_if_no_google


@pytest.fixture(scope="module")
def setup_pydantic_ai():
    """Set up PydanticAI with instrumentation."""
    if not config.has_google():
        pytest.skip("GOOGLE_API_KEY not set")

    from fi_instrumentation import register
    try:
        from traceai_pydantic_ai import PydanticAIInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_pydantic_ai not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    PydanticAIInstrumentor().instrument(tracer_provider=tracer_provider)

    yield

    PydanticAIInstrumentor().uninstrument()


@skip_if_no_google
class TestPydanticAIAgent:
    """Test PydanticAI agent instrumentation."""

    def test_simple_agent(self, setup_pydantic_ai):
        """Test simple agent run."""
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIModel

        model = OpenAIModel(
            config.google_model,
            base_url=config.google_openai_base_url,
            api_key=config.google_api_key,
        )

        agent = Agent(
            model,
            system_prompt="You are a helpful assistant. Answer briefly.",
        )

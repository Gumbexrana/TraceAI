"""
E2E Tests for OpenAI Agents SDK Instrumentation

Tests OpenAI Agents SDK instrumentation. Attempts Google endpoint first,
falls back to requiring OPENAI_API_KEY.
"""

import pytest
import time

from config import config, skip_if_no_google, skip_if_no_openai


@pytest.fixture(scope="module")
def setup_openai_agents():
    """Set up OpenAI Agents with instrumentation."""
    if not config.has_google() and not config.has_openai():
        pytest.skip("GOOGLE_API_KEY or OPENAI_API_KEY not set")

    from fi_instrumentation import register
    try:
        from traceai_openai_agents import OpenAIAgentsInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_openai_agents not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    try:
        OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)
    except (ImportError, AttributeError) as e:
        pytest.skip(f"traceai_openai_agents incompatible: {e}")

    yield

    try:
        OpenAIAgentsInstrumentor().uninstrument()
    except Exception:
        pass


@pytest.mark.skipif(
    not (config.has_google() or config.has_openai()),
    reason="GOOGLE_API_KEY or OPENAI_API_KEY required"
)
class TestOpenAIAgentsBasic:
    """Test OpenAI Agents SDK basic operations."""

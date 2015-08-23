"""
E2E Tests for Google ADK (Agent Development Kit) Instrumentation

Tests Google ADK instrumentation using GOOGLE_API_KEY directly.
"""

import pytest
import time

from config import config, skip_if_no_google


@pytest.fixture(scope="module")
def setup_google_adk():
    """Set up Google ADK with instrumentation."""
    if not config.has_google():
        pytest.skip("GOOGLE_API_KEY not set")

    from fi_instrumentation import register
    try:
        from traceai_google_adk import GoogleADKInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_google_adk not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)

    yield

    GoogleADKInstrumentor().uninstrument()


@skip_if_no_google
class TestGoogleADKAgent:
    """Test Google ADK agent operations."""

    def test_simple_agent(self, setup_google_adk):
        """Test simple ADK agent."""
        import os
        os.environ["GOOGLE_API_KEY"] = config.google_api_key

        from google.adk.agents import Agent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService

        agent = Agent(
            name="assistant",
            model=config.google_model,

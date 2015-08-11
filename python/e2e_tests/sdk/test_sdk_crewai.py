"""
E2E Tests for CrewAI SDK Instrumentation

Tests CrewAI instrumentation using Google's OpenAI-compatible endpoint.
"""

import pytest
import os
import time

from config import config, skip_if_no_google


@pytest.fixture(scope="module")
def setup_crewai():
    """Set up CrewAI with instrumentation."""
    if not config.has_google():
        pytest.skip("GOOGLE_API_KEY not set")

    from fi_instrumentation import register
    try:
        from traceai_crewai import CrewAIInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_crewai not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)

    # Set env vars for CrewAI LLM config
    os.environ["OPENAI_API_BASE"] = config.google_openai_base_url
    os.environ["OPENAI_API_KEY"] = config.google_api_key

    yield

    CrewAIInstrumentor().uninstrument()


@skip_if_no_google
class TestCrewAIExecution:
    """Test CrewAI execution."""

    def test_simple_crew(self, setup_crewai):
        """Test a simple crew with one agent and one task."""
        from crewai import Agent, Task, Crew

        agent = Agent(
            role="Helpful Assistant",
            goal="Answer questions briefly and accurately",
            backstory="You are a helpful AI assistant.",
            llm=f"openai/{config.google_model}",
            verbose=False,
        )

        task = Task(
            description="What is 2+2? Answer with just the number.",
            expected_output="A single number",
            agent=agent,
        )

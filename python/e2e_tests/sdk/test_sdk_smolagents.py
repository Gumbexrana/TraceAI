"""
E2E Tests for Smolagents SDK Instrumentation

Tests Smolagents (HuggingFace) instrumentation using Google's OpenAI-compatible endpoint.
"""

import pytest
import time

from config import config, skip_if_no_google


@pytest.fixture(scope="module")
def setup_smolagents():
    """Set up Smolagents with instrumentation."""
    if not config.has_google():
        pytest.skip("GOOGLE_API_KEY not set")

    from fi_instrumentation import register
    try:
        from traceai_smolagents import SmolagentsInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_smolagents not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    SmolagentsInstrumentor().instrument(tracer_provider=tracer_provider)

    yield

    SmolagentsInstrumentor().uninstrument()



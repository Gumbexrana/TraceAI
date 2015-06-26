"""
E2E Tests for Agno SDK Instrumentation

Tests Agno (OpenInference-based) instrumentation.
Agno agents can use OpenAI-compat LLMs; tries Google endpoint.
"""

import pytest
import time

from config import config, skip_if_no_google


@pytest.fixture(scope="module")
def setup_agno():
    """Set up Agno with instrumentation."""
    if not config.has_google():
        pytest.skip("GOOGLE_API_KEY not set")

    from fi_instrumentation import register
    try:
        from traceai_agno import AgnoInstrumentorWrapper
    except (ImportError, AttributeError):
        pytest.skip("traceai_agno not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    AgnoInstrumentorWrapper().instrument(tracer_provider=tracer_provider)

    yield

    AgnoInstrumentorWrapper().uninstrument()


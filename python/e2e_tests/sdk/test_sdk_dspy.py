"""
E2E Tests for DSPy SDK Instrumentation

Tests DSPy instrumentation using Google's OpenAI-compatible endpoint.
"""

import pytest
import time

from config import config, skip_if_no_google


@pytest.fixture(scope="module")
def setup_dspy():
    """Set up DSPy with instrumentation."""
    if not config.has_google():
        pytest.skip("GOOGLE_API_KEY not set")

    from fi_instrumentation import register
    try:
        from traceai_dspy import DSPyInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_dspy not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    DSPyInstrumentor().instrument(tracer_provider=tracer_provider)

    import dspy

    lm = dspy.LM(
        f"openai/{config.google_model}",
        api_base=config.google_openai_base_url,
        api_key=config.google_api_key,
    )
    dspy.configure(lm=lm)

    yield

    DSPyInstrumentor().uninstrument()


@skip_if_no_google
class TestDSPyPredict:
    """Test DSPy Predict module."""

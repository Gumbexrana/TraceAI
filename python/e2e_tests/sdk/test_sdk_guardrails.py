"""
E2E Tests for Guardrails AI SDK Instrumentation

Tests Guardrails AI instrumentation using litellm's gemini provider.
"""

import os
import pytest
import time

from config import config, skip_if_no_google


@pytest.fixture(scope="module")
def setup_guardrails():
    """Set up Guardrails AI with instrumentation."""
    if not config.has_google():
        pytest.skip("GOOGLE_API_KEY not set")

    # Guardrails 0.5+ uses litellm internally; set GEMINI_API_KEY for routing
    os.environ.setdefault("GEMINI_API_KEY", config.google_api_key)

    from fi_instrumentation import register
    try:
        from traceai_guardrails import GuardrailsInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_guardrails not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

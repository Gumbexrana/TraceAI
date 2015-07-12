"""
E2E Tests for AutoGen v0.4+ SDK Instrumentation

Tests AutoGen AgentChat instrumentation using Google's Gemini model via litellm.
"""

import os
import pytest
import asyncio
import time

from config import config, skip_if_no_google


@pytest.fixture(scope="module")
def setup_autogen():
    """Set up AutoGen with instrumentation."""
    if not config.has_google():
        pytest.skip("GOOGLE_API_KEY not set")

    os.environ.setdefault("GEMINI_API_KEY", config.google_api_key)

    from fi_instrumentation import register
    try:
        from traceai_autogen import AutogenInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_autogen not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,

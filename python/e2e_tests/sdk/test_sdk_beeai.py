"""
E2E Tests for BeeAI SDK Instrumentation

Tests BeeAI (IBM) instrumentation. Requires BeeAI/OpenInference packages.
Attempts to use Google endpoint via OpenAI-compat where possible.
"""

import pytest
import time

from config import config, skip_if_no_google


@pytest.fixture(scope="module")
def setup_beeai():
    """Set up BeeAI with instrumentation."""
    if not config.has_google():
        pytest.skip("GOOGLE_API_KEY not set")

    try:
        from fi_instrumentation import register
        from traceai_beeai import BeeAIInstrumentorWrapper

        tracer_provider = register(
            project_name=config.project_name,
            project_version_name=config.project_version_name,

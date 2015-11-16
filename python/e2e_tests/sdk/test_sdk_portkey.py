"""
E2E Tests for Portkey SDK Instrumentation

Tests Portkey instrumentation. Requires PORTKEY_API_KEY.
"""

import pytest
import time

from config import config, skip_if_no_portkey, skip_if_no_google


@pytest.fixture(scope="module")
def portkey_client():
    """Create an instrumented Portkey client."""
    if not config.has_portkey():
        pytest.skip("PORTKEY_API_KEY not set")

    from fi_instrumentation import register
    try:
        from traceai_portkey import PortkeyInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_portkey not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    PortkeyInstrumentor().instrument(tracer_provider=tracer_provider)

    from portkey_ai import Portkey

    client = Portkey(api_key=config.portkey_api_key)

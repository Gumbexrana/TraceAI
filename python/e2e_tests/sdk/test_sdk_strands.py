"""
E2E Tests for AWS Strands SDK Instrumentation

Tests Strands agents instrumentation. Requires AWS credentials for Bedrock models.
"""

import pytest
import time

from config import config, skip_if_no_bedrock


@pytest.fixture(scope="module")
def setup_strands():
    """Set up Strands with instrumentation."""
    if not config.has_bedrock():
        pytest.skip("AWS credentials not set")

    from fi_instrumentation import register
    try:
        from traceai_strands import StrandsInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_strands not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    StrandsInstrumentor().instrument(tracer_provider=tracer_provider)

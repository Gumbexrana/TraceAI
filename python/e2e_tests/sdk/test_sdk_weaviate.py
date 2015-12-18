"""
E2E Tests for Weaviate SDK Instrumentation

Tests Weaviate instrumentation. Requires a running Weaviate instance.
Set WEAVIATE_URL or run: docker run -p 8080:8080 semitechnologies/weaviate
"""

import pytest
import os
import time
import uuid

from config import config


WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
HAS_WEAVIATE = bool(os.getenv("WEAVIATE_URL"))

skip_if_no_weaviate = pytest.mark.skipif(
    not HAS_WEAVIATE, reason="WEAVIATE_URL not set"
)


@pytest.fixture(scope="module")
def weaviate_client():
    """Create an instrumented Weaviate client."""
    if not HAS_WEAVIATE:
        pytest.skip("WEAVIATE_URL not set")

    from fi_instrumentation import register
    try:
        from traceai_weaviate import WeaviateInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_weaviate not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    WeaviateInstrumentor().instrument(tracer_provider=tracer_provider)

    import weaviate

    client = weaviate.connect_to_local()

    yield client

    client.close()
    WeaviateInstrumentor().uninstrument()



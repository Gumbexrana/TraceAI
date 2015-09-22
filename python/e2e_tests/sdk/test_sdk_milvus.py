"""
E2E Tests for Milvus SDK Instrumentation

Tests Milvus instrumentation. Requires a running Milvus instance.
Set MILVUS_HOST or run: docker compose up -d milvus-standalone
"""

import pytest
import os
import time
import uuid

from config import config


MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
HAS_MILVUS = bool(os.getenv("MILVUS_HOST"))

skip_if_no_milvus = pytest.mark.skipif(
    not HAS_MILVUS, reason="MILVUS_HOST not set"
)


@pytest.fixture(scope="module")
def milvus_client():
    """Create an instrumented Milvus client."""
    if not HAS_MILVUS:
        pytest.skip("MILVUS_HOST not set")

    from fi_instrumentation import register
    try:
        from traceai_milvus import MilvusInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_milvus not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    MilvusInstrumentor().instrument(tracer_provider=tracer_provider)

    from pymilvus import MilvusClient

    client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")

    yield client

    MilvusInstrumentor().uninstrument()


@skip_if_no_milvus
class TestMilvusOperations:

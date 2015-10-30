"""
E2E Tests for Pinecone SDK Instrumentation

Tests Pinecone instrumentation. Requires PINECONE_API_KEY.
"""

import pytest
import time
import uuid

from config import config, skip_if_no_pinecone


@pytest.fixture(scope="module")
def pinecone_index():
    """Create an instrumented Pinecone index."""
    if not config.has_pinecone():
        pytest.skip("PINECONE_API_KEY not set")

    from fi_instrumentation import register
    try:
        from traceai_pinecone import PineconeInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_pinecone not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    PineconeInstrumentor().instrument(tracer_provider=tracer_provider)

    from pinecone import Pinecone

    pc = Pinecone(api_key=config.pinecone_api_key)

    # Use existing index or create serverless
    index_name = "e2e-test-index"
    existing = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing:
        from pinecone import ServerlessSpec

        pc.create_index(
            name=index_name,
            dimension=128,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to be ready
        import time as t
        t.sleep(10)

    index = pc.Index(index_name)

    yield index

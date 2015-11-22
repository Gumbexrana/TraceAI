"""
E2E Tests for Qdrant SDK Instrumentation

Tests Qdrant instrumentation with in-memory client. No API keys needed.
"""

import pytest
import time
import uuid

from config import config


@pytest.fixture(scope="module")
def qdrant_setup():
    """Set up Qdrant with instrumentation."""
    from fi_instrumentation import register
    try:
        from traceai_qdrant import QdrantInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_qdrant not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    QdrantInstrumentor().instrument(tracer_provider=tracer_provider)

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        pytest.skip("qdrant-client not installed")

    client = QdrantClient(":memory:")

    yield client

    QdrantInstrumentor().uninstrument()


@pytest.fixture
def collection_name():
    """Generate unique collection name."""
    return f"test_{uuid.uuid4().hex[:8]}"


class TestQdrantOperations:
    """Test Qdrant CRUD operations."""

    def test_create_collection_and_search(self, qdrant_setup, collection_name):
        """Test creating a collection, upserting points, and searching."""
        import random
        from qdrant_client.models import Distance, VectorParams, PointStruct

        qdrant_setup.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=128, distance=Distance.COSINE),
        )

        points = [
            PointStruct(
                id=i,
                vector=[random.uniform(-1, 1) for _ in range(128)],
                payload={"text": f"document {i}", "category": "A" if i < 5 else "B"},
            )
            for i in range(10)
        ]

        qdrant_setup.upsert(
            collection_name=collection_name,
            points=points,
        )

        results = qdrant_setup.query_points(
            collection_name=collection_name,
            query=points[0].vector,
            limit=3,
        )

        assert len(results.points) == 3
        assert results.points[0].id == 0  # Closest to itself
        time.sleep(2)
        print(f"Search results: {[r.id for r in results.points]}")

    def test_upsert_and_get(self, qdrant_setup, collection_name):
        """Test upsert and get by ID."""
        import random
        from qdrant_client.models import Distance, VectorParams, PointStruct

        qdrant_setup.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=128, distance=Distance.COSINE),
        )

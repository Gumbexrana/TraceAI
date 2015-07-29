"""
E2E Tests for ChromaDB SDK Instrumentation

Tests ChromaDB instrumentation with in-memory client. No API keys needed.
"""

import pytest
import time

from config import config


@pytest.fixture(scope="module")
def chromadb_setup():
    """Set up ChromaDB with instrumentation."""
    from fi_instrumentation import register
    try:
        from traceai_chromadb import ChromaDBInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_chromadb not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    ChromaDBInstrumentor().instrument(tracer_provider=tracer_provider)

    try:
        import chromadb
    except ImportError:
        pytest.skip("chromadb not installed")

    client = chromadb.Client()

    yield client

    ChromaDBInstrumentor().uninstrument()


@pytest.fixture
def collection(chromadb_setup):
    """Create a fresh collection for each test."""
    import uuid

    collection_name = f"test_{uuid.uuid4().hex[:8]}"
    collection = chromadb_setup.create_collection(name=collection_name)
    yield collection
    chromadb_setup.delete_collection(name=collection_name)


class TestChromaDBOperations:
    """Test ChromaDB CRUD operations."""

    def test_add_and_query(self, collection):
        """Test adding documents and querying."""
        collection.add(
            ids=["1", "2", "3"],
            documents=[
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subset of artificial intelligence.",
                "Python is a popular programming language.",
            ],
        )

        results = collection.query(
            query_texts=["programming language"],
            n_results=1,
        )

        assert results["ids"][0][0] == "3"
        time.sleep(2)
        print(f"Query result: {results['ids']}")

    def test_add_with_embeddings(self, collection):
        """Test adding documents with pre-computed embeddings."""
        import random

        embeddings = [[random.uniform(-1, 1) for _ in range(128)] for _ in range(3)]


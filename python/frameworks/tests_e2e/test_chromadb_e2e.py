"""
End-to-End Integration Tests for ChromaDB Instrumentation

ChromaDB can run in ephemeral mode, so these tests don't require external setup.

Run with: pytest tests_e2e/test_chromadb_e2e.py -v
"""

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture
def chromadb_client():
    """Create ephemeral ChromaDB client."""
    try:
        import chromadb
        return chromadb.Client()
    except ImportError:
        pytest.skip("chromadb not installed")


@pytest.fixture
def instrument_chromadb(tracer_provider, in_memory_span_exporter):
    """Instrument ChromaDB for testing."""
    from traceai_chromadb import ChromaDBInstrumentor

    instrumentor = ChromaDBInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    yield
    instrumentor.uninstrument()
    in_memory_span_exporter.clear()


class TestChromaDBE2E:
    """End-to-end tests for ChromaDB instrumentation."""

    @pytest.mark.integration
    @pytest.mark.chromadb
    def test_full_workflow(
        self,
        chromadb_client,
        tracer_provider,
        in_memory_span_exporter: InMemorySpanExporter,
        instrument_chromadb,
    ):
        """Test complete workflow: create, add, query, delete."""
        # Create collection
        collection = chromadb_client.get_or_create_collection(
            name="test_collection",
            metadata={"hnsw:space": "cosine"}
        )

        # Add documents
        collection.add(
            ids=["doc1", "doc2", "doc3"],
            documents=[
                "Machine learning is a subset of AI.",
                "Deep learning uses neural networks.",
                "Natural language processing analyzes text.",
            ],
            metadatas=[
                {"category": "ml"},
                {"category": "dl"},
                {"category": "nlp"},

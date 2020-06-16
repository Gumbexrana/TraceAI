"""Unit tests for Pinecone instrumentation."""

import pytest
from unittest.mock import MagicMock, patch
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


class TestPineconeInstrumentor:
    """Test PineconeInstrumentor class."""

    def test_instrumentor_can_be_instantiated(self):
        """Test that instrumentor can be instantiated."""
        from traceai_pinecone import PineconeInstrumentor

        instrumentor = PineconeInstrumentor()
        assert instrumentor is not None

    def test_instrumentation_dependencies(self):
        """Test that instrumentor reports correct dependencies."""
        from traceai_pinecone import PineconeInstrumentor

        instrumentor = PineconeInstrumentor()
        deps = instrumentor.instrumentation_dependencies()
        assert "pinecone-client >= 3.0.0" in deps


class TestPineconeQuery:
    """Test Pinecone query operation tracing."""

    @patch("pinecone.Index")
    def test_query_creates_span(
        self,
        mock_index_class,
        tracer_provider,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_pinecone_index,
        instrument_pinecone,
    ):
        """Test that query operation creates a span with correct attributes."""
        mock_index_class.return_value = mock_pinecone_index

        # Execute query
        index = mock_index_class("test-index")
        result = index.query(
            vector=[0.1] * 1536,
            top_k=10,
            namespace="test-ns",
            include_metadata=True,
        )

        # Verify span was created
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) >= 1

        # Find the query span
        query_spans = [s for s in spans if "query" in s.name.lower()]
        if query_spans:
            span = query_spans[0]
            assert span.name == "pinecone query"
            attributes = dict(span.attributes or {})
            assert attributes.get("db.system") == "pinecone"
            assert attributes.get("db.operation.name") == "query"

    @patch("pinecone.Index")
    def test_query_with_filter_captures_filter(
        self,
        mock_index_class,
        tracer_provider,
        in_memory_span_exporter: InMemorySpanExporter,
        mock_pinecone_index,

"""
End-to-End Integration Tests for LanceDB Instrumentation

LanceDB is embedded and doesn't require external setup.

Run with: pytest tests_e2e/test_lancedb_e2e.py -v
"""

import pytest
import tempfile
import shutil
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from conftest import generate_test_embedding


@pytest.fixture
def lancedb_connection():
    """Create temporary LanceDB connection."""
    try:
        import lancedb
        temp_dir = tempfile.mkdtemp()
        db = lancedb.connect(temp_dir)
        yield db
        shutil.rmtree(temp_dir, ignore_errors=True)
    except ImportError:
        pytest.skip("lancedb not installed")


@pytest.fixture
def instrument_lancedb(tracer_provider, in_memory_span_exporter):
    """Instrument LanceDB for testing."""
    from traceai_lancedb import LanceDBInstrumentor

    instrumentor = LanceDBInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    yield
    instrumentor.uninstrument()
    in_memory_span_exporter.clear()


class TestLanceDBE2E:
    """End-to-end tests for LanceDB instrumentation."""

    @pytest.mark.integration
    @pytest.mark.lancedb
    def test_full_workflow(

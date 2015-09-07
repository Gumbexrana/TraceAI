"""
E2E Tests for LanceDB SDK Instrumentation

Tests LanceDB instrumentation with local filesystem. No API keys needed.
"""

import pytest
import time
import tempfile
import shutil

from config import config


@pytest.fixture(scope="module")
def lancedb_setup():
    """Set up LanceDB with instrumentation."""
    from fi_instrumentation import register
    try:
        from traceai_lancedb import LanceDBInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_lancedb not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    LanceDBInstrumentor().instrument(tracer_provider=tracer_provider)

    # Create temp directory for LanceDB
    tmp_dir = tempfile.mkdtemp(prefix="lancedb_e2e_")

    try:
        import lancedb
    except ImportError:
        pytest.skip("lancedb not installed")

    db = lancedb.connect(tmp_dir)

    yield db

    LanceDBInstrumentor().uninstrument()
    shutil.rmtree(tmp_dir, ignore_errors=True)


class TestLanceDBOperations:
    """Test LanceDB CRUD operations."""

    def test_create_table_and_search(self, lancedb_setup):
        """Test creating a table and searching."""
        import random

        data = [

"""
E2E Tests for MongoDB Vector SDK Instrumentation

Tests MongoDB vector search instrumentation. Requires MongoDB Atlas or local MongoDB 7+.
Set MONGODB_URI to enable.
"""

import pytest
import os
import time
import uuid

from config import config


MONGODB_URI = os.getenv("MONGODB_URI")
HAS_MONGODB = bool(MONGODB_URI)

skip_if_no_mongodb = pytest.mark.skipif(
    not HAS_MONGODB, reason="MONGODB_URI not set"
)


@pytest.fixture(scope="module")
def mongodb_collection():
    """Create an instrumented MongoDB collection."""
    if not HAS_MONGODB:
        pytest.skip("MONGODB_URI not set")

    from fi_instrumentation import register
    try:
        from traceai_mongodb import MongoDBInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_mongodb not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    MongoDBInstrumentor().instrument(tracer_provider=tracer_provider)

    from pymongo import MongoClient

    client = MongoClient(MONGODB_URI)
    db = client["e2e_test_db"]
    collection_name = f"test_{uuid.uuid4().hex[:8]}"
    collection = db[collection_name]

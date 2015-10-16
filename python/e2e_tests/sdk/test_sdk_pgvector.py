"""
E2E Tests for pgvector SDK Instrumentation

Tests pgvector instrumentation. Requires PostgreSQL with pgvector extension.
"""

import pytest
import os
import time

from config import config


PG_VECTOR_HOST = os.getenv("PG_VECTOR_HOST", config.pg_host)
PG_VECTOR_PORT = os.getenv("PG_VECTOR_PORT", config.pg_port)
PG_VECTOR_DB = os.getenv("PG_VECTOR_DB", "pgvector_test")
PG_VECTOR_USER = os.getenv("PG_VECTOR_USER", config.pg_user)
PG_VECTOR_PASSWORD = os.getenv("PG_VECTOR_PASSWORD", config.pg_password)
HAS_PGVECTOR = bool(os.getenv("PG_VECTOR_HOST"))

skip_if_no_pgvector = pytest.mark.skipif(
    not HAS_PGVECTOR, reason="PG_VECTOR_HOST not set"
)


@pytest.fixture(scope="module")
def pgvector_conn():
    """Create an instrumented pgvector connection."""
    if not HAS_PGVECTOR:
        pytest.skip("PG_VECTOR_HOST not set")

    from fi_instrumentation import register
    try:
        from traceai_pgvector import PgVectorInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_pgvector not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    PgVectorInstrumentor().instrument(tracer_provider=tracer_provider)


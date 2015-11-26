"""
E2E Tests for Redis Vector SDK Instrumentation

Tests Redis vector search instrumentation. Requires Redis with RediSearch module.
Set REDIS_URL or run: docker run -p 6379:6379 redis/redis-stack
"""

import pytest
import os
import time
import uuid

from config import config


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
HAS_REDIS = bool(os.getenv("REDIS_URL"))

skip_if_no_redis = pytest.mark.skipif(
    not HAS_REDIS, reason="REDIS_URL not set"
)


@pytest.fixture(scope="module")
def redis_client():
    """Create an instrumented Redis client."""
    if not HAS_REDIS:
        pytest.skip("REDIS_URL not set")

    from fi_instrumentation import register
    try:
        from traceai_redis import RedisInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_redis not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    RedisInstrumentor().instrument(tracer_provider=tracer_provider)

    import redis


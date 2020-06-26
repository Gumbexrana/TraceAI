"""
TraceAI Pinecone Instrumentation

Provides OpenTelemetry instrumentation for Pinecone vector database operations.

Usage:
    from fi_instrumentation import register
    from traceai_pinecone import PineconeInstrumentor

    trace_provider = register(project_name="my-rag-app")
    PineconeInstrumentor().instrument(tracer_provider=trace_provider)

    # Now all Pinecone operations are automatically traced
    import pinecone
    pc = pinecone.Pinecone(api_key="...")
    index = pc.Index("my-index")
    results = index.query(vector=[0.1] * 1536, top_k=10)
"""

import logging
from importlib import import_module
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from wrapt import wrap_function_wrapper

from traceai_pinecone.version import __version__
from traceai_pinecone.package import _instruments
from traceai_pinecone._wrappers import (
    QueryWrapper,
    UpsertWrapper,
    DeleteWrapper,
    FetchWrapper,
    UpdateWrapper,
    DescribeIndexStatsWrapper,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_MODULE = "pinecone"


class PineconeInstrumentor(BaseInstrumentor):
    """
    OpenTelemetry instrumentor for Pinecone vector database.

    Instruments the following operations:
    - query: Vector similarity search
    - upsert: Insert or update vectors
    - delete: Delete vectors
    - fetch: Retrieve vectors by ID
    - update: Update vector metadata
    - describe_index_stats: Get index statistics
    """

    __slots__ = (
        "_original_query",
        "_original_upsert",
        "_original_delete",
        "_original_fetch",

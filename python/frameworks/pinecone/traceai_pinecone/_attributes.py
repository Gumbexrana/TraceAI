"""
Attribute constants for Pinecone vector database instrumentation.

Follows OpenTelemetry semantic conventions for databases with vector-specific extensions.
"""

import json
from typing import Any, Dict, Optional


class VectorDBAttributes:
    """
    Semantic conventions for vector database operations.

    Based on OpenTelemetry database semantic conventions with extensions
    for vector-specific operations.
    """

    # Core DB attributes (OTEL standard)
    DB_SYSTEM = "db.system"
    DB_OPERATION_NAME = "db.operation.name"
    DB_NAMESPACE = "db.namespace"

    # Query attributes
    QUERY_TOP_K = "db.vector.query.top_k"
    QUERY_FILTER = "db.vector.query.filter"
    QUERY_INCLUDE_METADATA = "db.vector.query.include_metadata"
    QUERY_INCLUDE_VECTORS = "db.vector.query.include_vectors"
    QUERY_SCORE_THRESHOLD = "db.vector.query.score_threshold"
    QUERY_METRIC = "db.vector.query.metric"

    # Result attributes
    RESULTS_COUNT = "db.vector.results.count"
    RESULTS_SCORES = "db.vector.results.scores"
    RESULTS_IDS = "db.vector.results.ids"

    # Upsert/Insert attributes
    UPSERT_COUNT = "db.vector.upsert.count"
    UPSERT_DIMENSIONS = "db.vector.upsert.dimensions"

    # Delete attributes
    DELETE_COUNT = "db.vector.delete.count"
    DELETE_ALL = "db.vector.delete.all"

    # Index/Collection attributes
    INDEX_NAME = "db.vector.index.name"
    INDEX_HOST = "db.vector.index.host"
    INDEX_METRIC = "db.vector.index.metric"
    INDEX_DIMENSIONS = "db.vector.index.dimensions"
    INDEX_TYPE = "db.vector.index.type"

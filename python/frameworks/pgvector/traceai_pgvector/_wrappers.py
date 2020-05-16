"""pgvector method wrappers for OpenTelemetry instrumentation."""

import re
import logging
from typing import Any, Callable, Optional, Tuple

from opentelemetry.trace import SpanKind, Status, StatusCode, Tracer

logger = logging.getLogger(__name__)

# Vector operators in pgvector
VECTOR_OPERATORS = {
    "<->": "l2_distance",
    "<=>": "cosine_distance",
    "<#>": "inner_product",
    "<+>": "l1_distance",
}

# Regex patterns for detecting vector operations
VECTOR_OP_PATTERN = re.compile(r"(<->|<=>|<#>|<\+>)")
ORDER_BY_PATTERN = re.compile(r"ORDER\s+BY\s+.*?(<->|<=>|<#>|<\+>)", re.IGNORECASE)
LIMIT_PATTERN = re.compile(r"LIMIT\s+(\d+)", re.IGNORECASE)
TABLE_PATTERN = re.compile(r"(?:FROM|INTO|UPDATE)\s+(\w+)", re.IGNORECASE)


def detect_vector_operation(query) -> Optional[Tuple[str, dict]]:
    """Detect if a query contains pgvector operations and extract metadata."""
    if not query:
        return None

    # Handle psycopg3 Composed objects and other non-string query types
    if hasattr(query, 'as_string'):
        try:
            # Composed.as_string() can work with None for simple cases
            query = query.as_string(None)
        except Exception:
            # Fall back to str() if as_string fails
            query = str(query)
    elif not isinstance(query, str):
        query = str(query)

    query_upper = query.upper()

    # Check for vector operators
    vector_match = VECTOR_OP_PATTERN.search(query)
    if not vector_match:
        return None

    operator = vector_match.group(1)
    distance_type = VECTOR_OPERATORS.get(operator, "unknown")

    metadata = {
        "db.vector.distance_type": distance_type,
        "db.vector.operator": operator,
    }

    # Check if it's a similarity search (ORDER BY with vector operator)
    order_match = ORDER_BY_PATTERN.search(query)
    if order_match:
        metadata["db.vector.query.type"] = "similarity_search"

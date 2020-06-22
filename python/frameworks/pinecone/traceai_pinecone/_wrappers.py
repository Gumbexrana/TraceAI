"""
Pinecone method wrappers for OpenTelemetry instrumentation.
"""

import json
import logging
from typing import Any, Callable, Optional

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode, Tracer

from traceai_pinecone._attributes import (
    VectorDBAttributes as Attrs,
    get_common_attributes,
    safe_json_dumps,
)

logger = logging.getLogger(__name__)


class BaseWrapper:
    """Base wrapper class for Pinecone operations."""

    def __init__(self, tracer: Tracer):
        self._tracer = tracer

    def _get_index_name(self, instance: Any) -> Optional[str]:
        """Extract index name from Pinecone Index instance."""
        try:
            # Pinecone v3+ stores index name in _config or as attribute
            if hasattr(instance, "_config"):
                return getattr(instance._config, "index_name", None)
            if hasattr(instance, "name"):
                return instance.name
            if hasattr(instance, "_index_name"):
                return instance._index_name
            # Try to get from host
            if hasattr(instance, "_config") and hasattr(instance._config, "host"):
                host = instance._config.host
                # Extract index name from host like "index-name-xxx.svc.pinecone.io"
                if host:
                    return host.split(".")[0].rsplit("-", 1)[0]
        except Exception:
            pass
        return "unknown"

    def _get_index_host(self, instance: Any) -> Optional[str]:
        """Extract index host from Pinecone Index instance."""
        try:
            if hasattr(instance, "_config") and hasattr(instance._config, "host"):
                return instance._config.host
        except Exception:
            pass
        return None


class QueryWrapper(BaseWrapper):
    """Wrapper for Pinecone Index.query() method."""

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        index_name = self._get_index_name(instance)
        namespace = kwargs.get("namespace", "")
        top_k = kwargs.get("top_k", 10)
        include_metadata = kwargs.get("include_metadata", False)
        include_values = kwargs.get("include_values", False)
        filter_dict = kwargs.get("filter")

        span_name = "pinecone query"

        attributes = get_common_attributes("pinecone", "query", index_name)
        attributes.update({
            Attrs.QUERY_TOP_K: top_k,
            Attrs.QUERY_INCLUDE_METADATA: include_metadata,
            Attrs.QUERY_INCLUDE_VECTORS: include_values,
            Attrs.NAMESPACE: namespace or "",
        })

        if filter_dict:
            attributes[Attrs.QUERY_FILTER] = safe_json_dumps(filter_dict)

        host = self._get_index_host(instance)
        if host:
            attributes[Attrs.INDEX_HOST] = host

        with self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.CLIENT,
            attributes=attributes,
        ) as span:
            try:
                result = wrapped(*args, **kwargs)

                # Extract results info
                if result and hasattr(result, "matches"):
                    matches = result.matches
                    span.set_attribute(Attrs.RESULTS_COUNT, len(matches))

                    # Get top scores
                    if matches:
                        scores = [m.score for m in matches[:10] if hasattr(m, "score")]
                        if scores:
                            span.set_attribute(Attrs.RESULTS_SCORES, safe_json_dumps(scores))

                        # Get result IDs
                        ids = [m.id for m in matches[:10] if hasattr(m, "id")]
                        if ids:
                            span.set_attribute(Attrs.RESULTS_IDS, safe_json_dumps(ids))

                span.set_status(Status(StatusCode.OK))
                return result

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise


class UpsertWrapper(BaseWrapper):
    """Wrapper for Pinecone Index.upsert() method."""

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        index_name = self._get_index_name(instance)
        namespace = kwargs.get("namespace", "")

        # Get vectors from args or kwargs
        vectors = args[0] if args else kwargs.get("vectors", [])
        vector_count = len(vectors) if vectors else 0

        # Try to get vector dimensions
        dimensions = None
        if vectors and len(vectors) > 0:
            first_vector = vectors[0]
            if isinstance(first_vector, dict) and "values" in first_vector:
                dimensions = len(first_vector["values"])
            elif hasattr(first_vector, "values"):
                dimensions = len(first_vector.values)

        span_name = "pinecone upsert"

        attributes = get_common_attributes("pinecone", "upsert", index_name)
        attributes.update({

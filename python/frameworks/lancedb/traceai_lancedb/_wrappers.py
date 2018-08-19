"""LanceDB method wrappers for OpenTelemetry instrumentation."""

import json
import logging
from typing import Any, Callable

from opentelemetry.trace import SpanKind, Status, StatusCode, Tracer

logger = logging.getLogger(__name__)


def safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj)
    except (TypeError, ValueError):
        return str(obj)


class BaseWrapper:
    def __init__(self, tracer: Tracer):
        self._tracer = tracer

    def _get_table_name(self, instance: Any) -> str:
        try:
            if hasattr(instance, "name"):
                return instance.name
            if hasattr(instance, "_name"):
                return instance._name
        except Exception:
            pass
        return "unknown"

    def _get_attributes(self, operation: str, table_name: str) -> dict:
        return {
            "db.system": "lancedb",
            "db.operation.name": operation,
            "db.namespace": table_name,
            "db.vector.collection.name": table_name,
        }


class SearchWrapper(BaseWrapper):
    def __init__(self, tracer: Tracer, method: str = "to_list"):
        super().__init__(tracer)
        self._method = method

    def __call__(self, wrapped: Callable, instance: Any, args: tuple, kwargs: dict) -> Any:
        table_name = self._get_table_name(instance)

        # Try to get limit from query builder
        limit = getattr(instance, "_limit", 10)

        attributes = self._get_attributes("search", table_name)
        attributes["db.vector.query.top_k"] = limit
        attributes["db.vector.search.output_format"] = self._method

        with self._tracer.start_as_current_span("lancedb search", kind=SpanKind.CLIENT, attributes=attributes) as span:
            try:
                result = wrapped(*args, **kwargs)
                if result is not None:
                    if self._method == "to_list":
                        span.set_attribute("db.vector.results.count", len(result))
                    elif hasattr(result, "num_rows"):
                        span.set_attribute("db.vector.results.count", result.num_rows)
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise



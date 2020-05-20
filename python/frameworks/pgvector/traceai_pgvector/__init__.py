"""
TraceAI pgvector Instrumentation

Provides OpenTelemetry instrumentation for pgvector PostgreSQL extension.
"""

import logging
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from wrapt import wrap_function_wrapper

from traceai_pgvector.version import __version__
from traceai_pgvector.package import _instruments
from traceai_pgvector._wrappers import (
    ExecuteWrapper,
    ExecuteManyWrapper,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class PgVectorInstrumentor(BaseInstrumentor):
    """OpenTelemetry instrumentor for pgvector PostgreSQL extension."""

    __slots__ = ("_originals",)

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()

        tracer = trace_api.get_tracer(__name__, __version__, tracer_provider)

        self._originals = {}

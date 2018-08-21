"""
TraceAI LanceDB Instrumentation

Provides OpenTelemetry instrumentation for LanceDB vector database operations.
"""

import logging
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from wrapt import wrap_function_wrapper

from traceai_lancedb.version import __version__
from traceai_lancedb.package import _instruments
from traceai_lancedb._wrappers import (
    SearchWrapper,
    AddWrapper,
    UpdateWrapper,
    DeleteWrapper,
    CreateTableWrapper,
    DropTableWrapper,
    OpenTableWrapper,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LanceDBInstrumentor(BaseInstrumentor):
    """OpenTelemetry instrumentor for LanceDB vector database."""

    __slots__ = ("_originals",)

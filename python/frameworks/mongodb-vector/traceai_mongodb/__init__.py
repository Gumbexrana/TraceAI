"""
TraceAI MongoDB Instrumentation

Provides OpenTelemetry instrumentation for MongoDB Atlas Vector Search operations.
"""

import logging
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from wrapt import wrap_function_wrapper

from traceai_mongodb.version import __version__
from traceai_mongodb.package import _instruments
from traceai_mongodb._wrappers import (
    AggregateWrapper,
    InsertOneWrapper,
    InsertManyWrapper,
    UpdateOneWrapper,
    UpdateManyWrapper,
    DeleteOneWrapper,
    DeleteManyWrapper,
    FindWrapper,
    FindOneWrapper,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())



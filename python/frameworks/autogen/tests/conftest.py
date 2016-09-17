"""Pytest configuration for AutoGen tests."""

import sys
from unittest.mock import MagicMock
from enum import Enum
import pytest

# Mock opentelemetry first
mock_otel = MagicMock()
mock_otel_trace = MagicMock()
mock_otel_instrumentation = MagicMock()
mock_otel_instrumentation_instrumentor = MagicMock()

# Create mock classes
class MockStatusCode(Enum):
    OK = 1
    ERROR = 2
    UNSET = 0

class MockSpanKind(Enum):
    INTERNAL = 0
    CLIENT = 1
    SERVER = 2
    PRODUCER = 3
    CONSUMER = 4

mock_otel_trace.StatusCode = MockStatusCode
mock_otel_trace.SpanKind = MockSpanKind
mock_otel_trace.Status = MagicMock
mock_otel_trace.Link = MagicMock
mock_otel_trace.SpanContext = MagicMock
mock_otel_trace.get_tracer = MagicMock(return_value=MagicMock())
mock_otel_trace.get_tracer_provider = MagicMock(return_value=MagicMock())
mock_otel_trace.get_current_span = MagicMock(return_value=MagicMock())
mock_otel_trace.set_span_in_context = MagicMock()

# Create BaseInstrumentor mock
class MockBaseInstrumentor:

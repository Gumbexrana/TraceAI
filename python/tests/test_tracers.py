"""Tests for fi_instrumentation.instrumentation._tracers module."""

import asyncio
import pytest
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock, Mock
from opentelemetry.trace import Tracer, Span, SpanKind
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

from fi_instrumentation.instrumentation._tracers import (
    FITracer,
    _IdGenerator,
    _infer_span_name,
    _infer_tool_description,
    _infer_tool_parameters,
    _get_jsonschema_type,
)
from fi_instrumentation.instrumentation.config import TraceConfig
from fi_instrumentation.fi_types import FiSpanKindValues

# Move this to module level to apply to all tests
@pytest.fixture(autouse=True)
def mock_trace_components():
    """Mock OpenTelemetry components for all tests in this module."""
    with patch('opentelemetry.trace.use_span') as mock_use_span, \
         patch('opentelemetry.context.get_value') as mock_get_value, \
         patch('opentelemetry.trace.Tracer.start_span') as mock_tracer_start_span:
        
        # Create a persistent mock span
        mock_span = MagicMock(spec=Span)
        mock_span.set_attribute = MagicMock()
        mock_span.set_attributes = MagicMock()
        mock_span.end = MagicMock()
        mock_span.is_recording.return_value = True
        mock_span.get_span_context.return_value = MagicMock()
        
        # Mock context values to not suppress instrumentation
        mock_get_value.return_value = False
        
        # Mock the Tracer class method that FITracer actually calls
        mock_tracer_start_span.return_value = mock_span
        
        # Mock use_span context manager
        context_manager = MagicMock()
        context_manager.__enter__ = MagicMock(return_value=mock_span)
        context_manager.__exit__ = MagicMock(return_value=None)
        mock_use_span.return_value = context_manager
        
        yield mock_span


class TestIdGenerator:
    """Test _IdGenerator functionality."""

    def test_id_generator_span_id_generation(self):
        """Test span ID generation."""
        generator = _IdGenerator()
        span_id = generator.generate_span_id()
        
        assert isinstance(span_id, int)
        assert span_id > 0
        assert span_id != 0  # Should not be INVALID_SPAN_ID
        
        # Should generate different IDs
        span_id2 = generator.generate_span_id()
        assert span_id != span_id2

    def test_id_generator_trace_id_generation(self):
        """Test trace ID generation."""
        generator = _IdGenerator()
        trace_id = generator.generate_trace_id()
        
        assert isinstance(trace_id, int)
        assert trace_id > 0
        assert trace_id != 0  # Should not be INVALID_TRACE_ID
        
        # Should generate different IDs
        trace_id2 = generator.generate_trace_id()
        assert trace_id != trace_id2

    def test_id_generator_avoids_invalid_ids(self):
        """Test that generator avoids invalid IDs."""
        generator = _IdGenerator()
        
        # Generate multiple IDs to ensure we don't get invalid ones
        for _ in range(100):
            span_id = generator.generate_span_id()
            trace_id = generator.generate_trace_id()
            
            assert span_id != 0  # INVALID_SPAN_ID
            assert trace_id != 0  # INVALID_TRACE_ID


class TestFITracer:
    """Test FITracer functionality."""

    @pytest.fixture
    def mock_tracer(self):
        """Create a mock OpenTelemetry tracer."""
        from opentelemetry.trace import Span
        
        tracer = MagicMock(spec=Tracer)
        tracer.id_generator = RandomIdGenerator()
        
        # Create a proper mock span that won't cause NoneType errors
        mock_span = MagicMock(spec=Span)
        mock_span.set_attribute = MagicMock()
        mock_span.set_attributes = MagicMock()
        mock_span.end = MagicMock()
        mock_span.is_recording.return_value = True
        mock_span.get_span_context.return_value = MagicMock()
        
        # Make sure start_span returns the mock span
        tracer.start_span.return_value = mock_span
        
        # Mock the context manager properly
        context_manager = MagicMock()
        context_manager.__enter__ = MagicMock(return_value=mock_span)
        context_manager.__exit__ = MagicMock(return_value=None)
        tracer.start_as_current_span.return_value = context_manager
        
        return tracer

    @pytest.fixture  
    def fi_tracer(self, mock_tracer):
        """Create a FITracer instance with mocked dependencies."""
        config = TraceConfig()
        return FITracer(mock_tracer, config)

    def test_fi_tracer_initialization(self, mock_tracer):
        """Test FITracer initialization."""
        config = TraceConfig()
        tracer = FITracer(mock_tracer, config)
        
        assert tracer._self_config == config
        assert isinstance(tracer._self_id_generator, _IdGenerator)

    def test_fi_tracer_id_generator_property(self, fi_tracer, mock_tracer):
        """Test id_generator property."""
        # When wrapped tracer has RandomIdGenerator, should return custom generator
        mock_tracer.id_generator = RandomIdGenerator()
        assert isinstance(fi_tracer.id_generator, _IdGenerator)
        
        # When wrapped tracer has other generator, should return that
        custom_generator = MagicMock()
        mock_tracer.id_generator = custom_generator
        assert fi_tracer.id_generator == custom_generator

    def test_fi_tracer_start_span_basic(self, fi_tracer, mock_tracer):
        """Test basic span creation."""
        # FITracer calls the class method, not instance method
        span = fi_tracer.start_span("test_span")
        
        assert span is not None
        # The span should be a FiSpan wrapper
        assert hasattr(span, 'set_attributes')
        assert hasattr(span, 'set_attribute')

    def test_fi_tracer_start_span_with_attributes(self, fi_tracer, mock_tracer):
        """Test span creation with attributes."""
        attributes = {"key1": "value1", "key2": "value2"}
        span = fi_tracer.start_span("test_span", attributes=attributes)
        
        assert span is not None
        # The span should be a FiSpan wrapper that handles attributes
        assert hasattr(span, 'set_attributes')
        # Since we have comprehensive mocking, this should work without errors

    def test_fi_tracer_start_as_current_span(self, fi_tracer, mock_tracer):
        """Test start_as_current_span context manager."""
        # Use the already configured mock tracer  
        with fi_tracer.start_as_current_span("test_span") as span:
            assert span is not None

    def test_fi_tracer_agent_decorator_sync(self, fi_tracer):
        """Test agent decorator on synchronous function."""
        @fi_tracer.agent
        def test_function(x: int, y: int) -> int:
            return x + y
        
        result = test_function(2, 3)
        assert result == 5

    def test_fi_tracer_agent_decorator_with_name(self, fi_tracer):
        """Test agent decorator with custom name."""
        @fi_tracer.agent(name="custom_agent")
        def test_function(x: int) -> int:

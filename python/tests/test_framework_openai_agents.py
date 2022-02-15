"""
Comprehensive test suite for OpenAI Agents instrumentation framework.
Tests the OpenAIAgentsInstrumentor and FiTracingProcessor functionality.
"""

import logging
import unittest
from unittest.mock import Mock, patch, MagicMock, ANY, call
from typing import Any, Collection

import pytest
from agents import Trace, TracingProcessor
from agents.tracing import Span
from agents.tracing.span_data import SpanData, ResponseSpanData, GenerationSpanData, FunctionSpanData

from fi_instrumentation import FITracer, TraceConfig
from opentelemetry import trace as trace_api
from opentelemetry.trace import Tracer, Status, StatusCode
from opentelemetry.trace.span import Span as OtelSpan
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.context import attach, detach
from opentelemetry.util.types import AttributeValue

from traceai_openai_agents import OpenAIAgentsInstrumentor
from traceai_openai_agents._processor import FiTracingProcessor


class TestOpenAIAgentsInstrumentor:
    """Test suite for OpenAIAgentsInstrumentor class."""

    def test_inheritance(self):
        """Test that OpenAIAgentsInstrumentor inherits from BaseInstrumentor."""
        instrumentor = OpenAIAgentsInstrumentor()
        assert isinstance(instrumentor, BaseInstrumentor)

    def test_instrumentation_dependencies(self):
        """Test that instrumentor returns correct dependencies."""
        instrumentor = OpenAIAgentsInstrumentor()
        dependencies = instrumentor.instrumentation_dependencies()
        assert isinstance(dependencies, Collection)
        assert "openai-agents >= 0.0.7" in dependencies

    @patch('traceai_openai_agents.trace_api.get_tracer_provider')
    @patch('traceai_openai_agents.trace_api.get_tracer')
    @patch('agents.set_trace_processors')
    def test_instrument_with_default_config(self, mock_set_processor, mock_get_tracer, mock_get_provider):
        """Test instrumentation with default configuration."""
        mock_provider = Mock()
        mock_tracer = Mock(spec=Tracer)
        mock_get_provider.return_value = mock_provider
        mock_get_tracer.return_value = mock_tracer

        instrumentor = OpenAIAgentsInstrumentor()
        instrumentor._instrument()

        mock_get_provider.assert_called_once()
        mock_get_tracer.assert_called_once()
        mock_set_processor.assert_called_once()
        
        # Verify FiTracingProcessor was created with correct tracer
        args, kwargs = mock_set_processor.call_args
        processor = args[0][0]
        assert isinstance(processor, FiTracingProcessor)

    @patch('traceai_openai_agents.trace_api.get_tracer')
    @patch('agents.set_trace_processors')
    def test_instrument_with_custom_tracer_provider(self, mock_set_processor, mock_get_tracer):
        """Test instrumentation with custom tracer provider."""
        mock_provider = Mock()
        mock_tracer = Mock(spec=Tracer)
        mock_get_tracer.return_value = mock_tracer

        instrumentor = OpenAIAgentsInstrumentor()
        instrumentor._instrument(tracer_provider=mock_provider)

        mock_get_tracer.assert_called_once()
        mock_set_processor.assert_called_once()

    @patch('traceai_openai_agents.trace_api.get_tracer_provider')
    @patch('traceai_openai_agents.trace_api.get_tracer')
    @patch('agents.set_trace_processors')
    def test_instrument_with_custom_config(self, mock_set_processor, mock_get_tracer, mock_get_provider):
        """Test instrumentation with custom trace config."""
        mock_provider = Mock()
        mock_tracer = Mock(spec=Tracer)
        mock_get_provider.return_value = mock_provider
        mock_get_tracer.return_value = mock_tracer
        
        custom_config = TraceConfig()
        instrumentor = OpenAIAgentsInstrumentor()
        instrumentor._instrument(config=custom_config)

        mock_set_processor.assert_called_once()

    @patch('traceai_openai_agents.trace_api.get_tracer_provider')
    @patch('traceai_openai_agents.trace_api.get_tracer')
    @patch('agents.set_trace_processors')
    def test_instrument_with_invalid_config_type(self, mock_set_processor, mock_get_tracer, mock_get_provider):
        """Test instrumentation fails with invalid config type."""
        mock_provider = Mock()
        mock_get_provider.return_value = mock_provider

        instrumentor = OpenAIAgentsInstrumentor()
        
        with pytest.raises(AssertionError):
            instrumentor._instrument(config="invalid_config")

    @patch('agents.set_trace_processors')
    def test_instrument_missing_agents_import(self, mock_set_processor):
        """Test instrumentation handles missing agents import gracefully."""
        mock_set_processor.side_effect = ImportError("No module named 'agents'")
        
        instrumentor = OpenAIAgentsInstrumentor()
        
        with pytest.raises(ImportError):
            instrumentor._instrument()

    @patch('traceai_openai_agents.trace_api.get_tracer_provider')
    @patch('agents.set_trace_processors')
    def test_instrument_exception_handling(self, mock_set_processor, mock_get_provider):
        """Test instrumentation handles exceptions properly."""
        mock_get_provider.side_effect = Exception("Test exception")
        
        instrumentor = OpenAIAgentsInstrumentor()
        
        with pytest.raises(Exception, match="Test exception"):
            instrumentor._instrument()

    def test_uninstrument_todo(self):
        """Test that uninstrument method exists but is not implemented."""
        instrumentor = OpenAIAgentsInstrumentor()
        # Should not raise an exception
        instrumentor._uninstrument()


class TestFiTracingProcessor:
    """Test suite for FiTracingProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_tracer = Mock(spec=Tracer)
        self.processor = FiTracingProcessor(self.mock_tracer)

    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor._tracer == self.mock_tracer
        assert isinstance(self.processor._root_spans, dict)
        assert isinstance(self.processor._otel_spans, dict)
        assert isinstance(self.processor._tokens, dict)

    def test_inheritance(self):
        """Test that FiTracingProcessor inherits from TracingProcessor."""
        assert isinstance(self.processor, TracingProcessor)

    def test_on_trace_start(self):
        """Test trace start handling."""
        mock_span = Mock(spec=OtelSpan)
        self.mock_tracer.start_span.return_value = mock_span
        
        trace = Mock(spec=Trace)
        trace.name = "test_trace"
        trace.trace_id = "trace_123"
        
        self.processor.on_trace_start(trace)
        
        self.mock_tracer.start_span.assert_called_once_with(
            name="test_trace",
            attributes={"fi.span.kind": "AGENT"}
        )
        assert self.processor._root_spans["trace_123"] == mock_span

    def test_on_trace_end(self):
        """Test trace end handling."""
        mock_span = Mock(spec=OtelSpan)
        trace_id = "trace_123"
        self.processor._root_spans[trace_id] = mock_span
        
        trace = Mock(spec=Trace)
        trace.trace_id = trace_id
        
        self.processor.on_trace_end(trace)
        
        mock_span.set_status.assert_called_once_with(ANY)
        mock_span.end.assert_called_once()
        assert trace_id not in self.processor._root_spans

    def test_on_trace_end_missing_span(self):
        """Test trace end handling when span doesn't exist."""
        trace = Mock(spec=Trace)
        trace.trace_id = "nonexistent_trace"
        
        # Should not raise an exception
        self.processor.on_trace_end(trace)

    @patch('traceai_openai_agents._processor._as_utc_nano')
    @patch('traceai_openai_agents._processor._get_span_name')
    @patch('traceai_openai_agents._processor._get_span_kind')
    @patch('traceai_openai_agents._processor.safe_json_dumps')
    @patch('traceai_openai_agents._processor.set_span_in_context')
    @patch('traceai_openai_agents._processor.attach')
    def test_on_span_start(self, mock_attach, mock_set_context, mock_json_dumps, 
                          mock_get_kind, mock_get_name, mock_utc_nano):
        """Test span start handling."""
        # Setup mocks
        mock_otel_span = Mock(spec=OtelSpan)
        self.mock_tracer.start_span.return_value = mock_otel_span
        mock_get_name.return_value = "test_span"
        mock_get_kind.return_value = "test_kind"
        mock_json_dumps.return_value = '{"test": "data"}'
        mock_utc_nano.return_value = 1234567890
        mock_context = Mock()
        mock_set_context.return_value = mock_context
        mock_token = Mock()
        mock_attach.return_value = mock_token
        
        # Create test span
        span = Mock()
        span.span_id = "span_123"
        span.trace_id = "trace_123"
        span.parent_id = None
        span.started_at = "2023-01-01T00:00:00"
        span.span_data = Mock(spec=SpanData)
        
        # Add root span
        root_span = Mock(spec=OtelSpan)
        self.processor._root_spans["trace_123"] = root_span
        
        self.processor.on_span_start(span)
        
        # Verify span creation
        self.mock_tracer.start_span.assert_called_once()
        assert self.processor._otel_spans["span_123"] == mock_otel_span
        assert self.processor._tokens["span_123"] == mock_token

    def test_on_span_start_no_timestamp(self):
        """Test span start handling when timestamp is missing."""
        span = Mock()
        span.started_at = None
        
        self.processor.on_span_start(span)
        
        # Should not call start_span if no timestamp
        self.mock_tracer.start_span.assert_not_called()

    @patch('traceai_openai_agents._processor.detach')
    @patch('traceai_openai_agents._processor._get_span_name')
    def test_on_span_end(self, mock_get_name, mock_detach):
        """Test span end handling."""
        # Setup
        span_id = "span_123"
        mock_token = Mock()
        mock_otel_span = Mock(spec=OtelSpan)
        mock_get_name.return_value = "test_span"
        
        self.processor._tokens[span_id] = mock_token
        self.processor._otel_spans[span_id] = mock_otel_span
        
        span = Mock()
        span.span_id = span_id
        span.ended_at = "2023-01-01T00:01:00"
        span.span_data = Mock(spec=SpanData)
        
        self.processor.on_span_end(span)

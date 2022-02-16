"""
Comprehensive test suite for Portkey instrumentation framework.
Tests the PortkeyInstrumentor and related wrapper functionality.
"""

import logging
import unittest
from unittest.mock import Mock, patch, MagicMock, ANY, call
from typing import Any, Collection, Dict, List, Mapping
from enum import Enum

import pytest
from portkey_ai.api_resources.apis.chat_complete import Completions

from fi_instrumentation import FITracer, TraceConfig
from opentelemetry import trace as trace_api
from opentelemetry.trace import Tracer, Status, StatusCode
from opentelemetry.trace.span import Span as OtelSpan
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.util.types import AttributeValue

from traceai_portkey import PortkeyInstrumentor
from traceai_portkey._wrappers import _CompletionsWrapper, _WithTracer, _flatten, _parse_args
from traceai_portkey._request_attributes_extractor import _RequestAttributesExtractor
from traceai_portkey._response_attributes_extractor import _ResponseAttributesExtractor
from traceai_portkey._utils import _io_value_and_type, _as_input_attributes, _as_output_attributes, _finish_tracing, _ValueAndType
from traceai_portkey._with_span import _WithSpan


class TestPortkeyInstrumentor:
    """Test suite for PortkeyInstrumentor class."""

    def test_inheritance(self):
        """Test that PortkeyInstrumentor inherits from BaseInstrumentor."""
        instrumentor = PortkeyInstrumentor()
        assert isinstance(instrumentor, BaseInstrumentor)

    def test_instrumentation_dependencies(self):
        """Test that instrumentor returns correct dependencies."""
        instrumentor = PortkeyInstrumentor()
        dependencies = instrumentor.instrumentation_dependencies()
        assert isinstance(dependencies, Collection)
        assert "portkey_ai >= 0.1.0" in dependencies

    @patch('traceai_portkey.trace_api.get_tracer_provider')
    @patch('traceai_portkey.trace_api.get_tracer')
    @patch('traceai_portkey.wrap_function_wrapper')
    def test_instrument_with_default_config(self, mock_wrap, mock_get_tracer, mock_get_provider):
        """Test instrumentation with default configuration."""
        mock_provider = Mock()
        mock_tracer = Mock(spec=Tracer)
        mock_get_provider.return_value = mock_provider
        mock_get_tracer.return_value = mock_tracer

        instrumentor = PortkeyInstrumentor()
        instrumentor._original_completions_create = None
        instrumentor._instrument()

        mock_get_provider.assert_called_once()
        mock_get_tracer.assert_called_once()
        assert mock_wrap.call_count == 2  # Two wrap calls

    @patch('traceai_portkey.trace_api.get_tracer')
    @patch('traceai_portkey.wrap_function_wrapper')
    def test_instrument_with_custom_tracer_provider(self, mock_wrap, mock_get_tracer):
        """Test instrumentation with custom tracer provider."""
        mock_provider = Mock()
        mock_tracer = Mock(spec=Tracer)
        mock_get_tracer.return_value = mock_tracer

        instrumentor = PortkeyInstrumentor()
        instrumentor._original_completions_create = None
        instrumentor._instrument(tracer_provider=mock_provider)

        mock_get_tracer.assert_called_once()
        assert mock_wrap.call_count == 2

    @patch('traceai_portkey.trace_api.get_tracer_provider')
    @patch('traceai_portkey.trace_api.get_tracer')
    @patch('traceai_portkey.wrap_function_wrapper')
    def test_instrument_with_custom_config(self, mock_wrap, mock_get_tracer, mock_get_provider):
        """Test instrumentation with custom trace config."""
        mock_provider = Mock()
        mock_tracer = Mock(spec=Tracer)
        mock_get_provider.return_value = mock_provider
        mock_get_tracer.return_value = mock_tracer
        
        custom_config = TraceConfig()
        instrumentor = PortkeyInstrumentor()
        instrumentor._original_completions_create = None
        instrumentor._instrument(config=custom_config)

        assert mock_wrap.call_count == 2

    @patch('traceai_portkey.trace_api.get_tracer_provider')
    @patch('traceai_portkey.trace_api.get_tracer')
    @patch('traceai_portkey.wrap_function_wrapper')
    def test_instrument_with_invalid_config_type(self, mock_wrap, mock_get_tracer, mock_get_provider):
        """Test instrumentation fails with invalid config type."""
        mock_provider = Mock()
        mock_get_provider.return_value = mock_provider

        instrumentor = PortkeyInstrumentor()
        
        with pytest.raises(AssertionError):
            instrumentor._instrument(config="invalid_config")

    @patch('traceai_portkey.import_module')
    def test_uninstrument(self, mock_import_module):
        """Test uninstrumentation process."""
        mock_module = Mock()
        mock_import_module.return_value = mock_module
        
        instrumentor = PortkeyInstrumentor()
        original_create = Mock()
        instrumentor._original_completions_create = original_create
        
        instrumentor._uninstrument()
        
        mock_import_module.assert_called_once_with("portkey_ai.api_resources.apis.chat_complete")
        assert mock_module.Completions.create == original_create

    @patch('traceai_portkey.import_module')
    def test_uninstrument_with_none_original(self, mock_import_module):
        """Test uninstrumentation when original function is None."""
        mock_module = Mock()
        mock_import_module.return_value = mock_module
        
        instrumentor = PortkeyInstrumentor()
        instrumentor._original_completions_create = None
        
        instrumentor._uninstrument()
        
        mock_import_module.assert_called_once()
        # Should not set attribute when original is None


class TestCompletionsWrapper:
    """Test suite for _CompletionsWrapper class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_tracer = Mock(spec=Tracer)
        self.wrapper = _CompletionsWrapper(tracer=self.mock_tracer)

    def test_initialization(self):
        """Test wrapper initialization."""
        assert self.wrapper._tracer == self.mock_tracer
        assert isinstance(self.wrapper._request_extractor, _RequestAttributesExtractor)
        assert isinstance(self.wrapper._response_extractor, _ResponseAttributesExtractor)

    def test_inheritance(self):
        """Test that _CompletionsWrapper inherits from _WithTracer."""
        assert isinstance(self.wrapper, _WithTracer)

    @patch('traceai_portkey._wrappers.context_api.get_value')
    def test_call_with_suppressed_instrumentation(self, mock_get_value):
        """Test wrapper when instrumentation is suppressed."""
        mock_get_value.return_value = True
        wrapped = Mock(return_value="response")
        
        result = self.wrapper(wrapped, None, (), {})
        
        assert result == "response"
        wrapped.assert_called_once_with()

    @patch('traceai_portkey._wrappers.context_api.get_value')
    @patch('traceai_portkey._wrappers._parse_args')
    @patch('traceai_portkey._wrappers.signature')
    @patch('traceai_portkey._wrappers.get_attributes_from_context')
    def test_call_successful_completion(self, mock_get_context, mock_signature, 
                                      mock_parse_args, mock_get_value):
        """Test successful completion call."""
        mock_get_value.return_value = False
        mock_parse_args.return_value = {"model": "gpt-3.5-turbo"}
        mock_signature.return_value = Mock()
        mock_get_context.return_value = []
        
        wrapped = Mock(return_value="response")
        
        # Mock the span context manager
        mock_span = Mock()
        self.wrapper._start_as_current_span = Mock()
        self.wrapper._start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
        self.wrapper._start_as_current_span.return_value.__exit__ = Mock(return_value=None)
        
        result = self.wrapper(wrapped, None, (), {"model": "gpt-3.5-turbo"})
        
        assert result == "response"
        wrapped.assert_called_once_with(model="gpt-3.5-turbo")

    @patch('traceai_portkey._wrappers.context_api.get_value')
    @patch('traceai_portkey._wrappers._parse_args')
    @patch('traceai_portkey._wrappers.signature')
    @patch('traceai_portkey._wrappers.get_attributes_from_context')
    def test_call_with_exception(self, mock_get_context, mock_signature, 
                                mock_parse_args, mock_get_value):
        """Test completion call that raises exception."""
        mock_get_value.return_value = False
        mock_parse_args.return_value = {"model": "gpt-3.5-turbo"}
        mock_signature.return_value = Mock()
        mock_get_context.return_value = []
        
        test_exception = Exception("Test error")
        wrapped = Mock(side_effect=test_exception)
        
        # Mock the span context manager
        mock_span = Mock()
        self.wrapper._start_as_current_span = Mock()
        self.wrapper._start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
        self.wrapper._start_as_current_span.return_value.__exit__ = Mock(return_value=None)
        
        with pytest.raises(Exception, match="Test error"):
            self.wrapper(wrapped, None, (), {"model": "gpt-3.5-turbo"})
        
        mock_span.record_exception.assert_called_once_with(test_exception)
        mock_span.finish_tracing.assert_called_once()


class TestWithTracer:
    """Test suite for _WithTracer base class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_tracer = Mock(spec=Tracer)
        self.with_tracer = _WithTracer(tracer=self.mock_tracer)

    def test_initialization(self):
        """Test base class initialization."""
        assert self.with_tracer._tracer == self.mock_tracer

    @patch('traceai_portkey._wrappers.trace_api.use_span')
    def test_start_as_current_span(self, mock_use_span):
        """Test span context manager creation."""
        mock_span = Mock(spec=OtelSpan)
        self.mock_tracer.start_span.return_value = mock_span
        mock_use_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_use_span.return_value.__exit__ = Mock(return_value=None)
        
        attributes = [("key1", "value1")]
        context_attributes = [("context_key", "context_value")]
        extra_attributes = [("extra_key", "extra_value")]
        
        with self.with_tracer._start_as_current_span(
            "test_span", attributes, context_attributes, extra_attributes
        ) as span:
            assert isinstance(span, _WithSpan)
        
        self.mock_tracer.start_span.assert_called_once_with(
            name="test_span", 
            attributes={"extra_key": "extra_value"}
        )

    @patch('traceai_portkey._wrappers.trace_api.use_span')
    @patch('traceai_portkey._wrappers.INVALID_SPAN')
    def test_start_as_current_span_with_exception(self, mock_invalid_span, mock_use_span):
        """Test span creation when tracer raises exception."""
        self.mock_tracer.start_span.side_effect = Exception("Span creation failed")
        mock_use_span.return_value.__enter__ = Mock(return_value=mock_invalid_span)
        mock_use_span.return_value.__exit__ = Mock(return_value=None)
        
        with self.with_tracer._start_as_current_span(
            "test_span", [], [], []
        ) as span:
            assert isinstance(span, _WithSpan)


class TestRequestAttributesExtractor:
    """Test suite for _RequestAttributesExtractor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = _RequestAttributesExtractor()

    def test_get_attributes_from_request(self):
        """Test basic request attribute extraction."""
        request_params = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello"}]}
        
        attributes = list(self.extractor.get_attributes_from_request(request_params))
        
        # Should contain span kind
        assert any(key == "fi.span.kind" for key, value in attributes)

    def test_get_extra_attributes_from_request_basic(self):
        """Test extra attributes extraction from request."""
        request_params = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        attributes = dict(self.extractor.get_extra_attributes_from_request(request_params))
        
        # Should contain invocation parameters
        assert "llm.invocation_parameters" in attributes

    def test_get_extra_attributes_with_tools(self):
        """Test extra attributes extraction with tools."""
        request_params = {
            "model": "gpt-3.5-turbo",
            "tools": [
                {"type": "function", "function": {"name": "test_tool"}},
                {"type": "function", "function": {"name": "test_tool2"}}
            ]
        }
        
        attributes = dict(self.extractor.get_extra_attributes_from_request(request_params))
        
        # Should contain tool schemas
        assert "llm.tools.0.tool.json_schema" in attributes
        assert "llm.tools.1.tool.json_schema" in attributes

    def test_get_extra_attributes_with_prompt_id(self):
        """Test extra attributes extraction with prompt ID."""
        request_params = {
            "model": "gpt-3.5-turbo",
            "prompt_id": "test_prompt_123"
        }
        
        attributes = dict(self.extractor.get_extra_attributes_from_request(request_params))
        
        assert "prompt.id" in attributes
        assert attributes["prompt.id"] == "test_prompt_123"

    def test_get_extra_attributes_with_variables(self):
        """Test extra attributes extraction with prompt variables."""
        request_params = {
            "model": "gpt-3.5-turbo",
            "variables": {"name": "Alice", "topic": "AI"}
        }
        
        attributes = dict(self.extractor.get_extra_attributes_from_request(request_params))
        
        assert "llm.prompt_template.variables" in attributes

    def test_get_extra_attributes_with_messages(self):
        """Test extra attributes extraction with input messages."""
        request_params = {
            "model": "gpt-3.5-turbo",
            "messages": [

"""
Comprehensive test suite for Vertex AI instrumentation framework.
Tests the VertexAIInstrumentor and related wrapper functionality.
"""

import asyncio
import logging
import unittest
from unittest.mock import Mock, patch, MagicMock, ANY, call
from typing import Any, Collection, Dict, List, Optional, AsyncIterator, Iterator
from collections.abc import AsyncIterable, Iterable

import pytest
import proto

from fi_instrumentation import FITracer, TraceConfig
from opentelemetry import trace as trace_api
from opentelemetry.trace import Tracer, Status, StatusCode
from opentelemetry.trace.span import Span as OtelSpan
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.util.types import AttributeValue

from traceai_vertexai import VertexAIInstrumentor
from traceai_vertexai._wrapper import (
    _Wrapper, _CallbackForAwaitable, _CallbackForIterable,
    _get_prediction_service_request, _update_span, _parse_usage_metadata,
    _parse_content, _parse_tool_calls, _parse_parts, _parse_part,
    _parse_predictions, _role, _extract_image_data, _finish,
    _GenerateContentResponseAccumulator, _NoOpResponseAccumulator
)
from traceai_vertexai._accumulator import (
    _KeyValuesAccumulator, _StringAccumulator, _IndexedAccumulator
)
from traceai_vertexai._proxy import _proxy


class TestVertexAIInstrumentor:
    """Test suite for VertexAIInstrumentor class."""

    def test_inheritance(self):
        """Test that VertexAIInstrumentor inherits from BaseInstrumentor."""
        instrumentor = VertexAIInstrumentor()
        assert isinstance(instrumentor, BaseInstrumentor)

    def test_instrumentation_dependencies(self):
        """Test that instrumentor returns correct dependencies."""
        instrumentor = VertexAIInstrumentor()
        dependencies = instrumentor.instrumentation_dependencies()
        assert isinstance(dependencies, Collection)
        assert "google-cloud-aiplatform >= 1.49.0" in dependencies

    def test_instrument_with_default_config(self):
        """Test instrumentation with default configuration."""
        with patch('traceai_vertexai.get_tracer_provider') as mock_get_provider, \
             patch('traceai_vertexai.get_tracer') as mock_get_tracer, \
             patch('traceai_vertexai.wrap_function_wrapper') as mock_wrap, \
             patch('google.api_core.gapic_v1') as mock_gapic:
            
            mock_provider = Mock()
            mock_tracer = Mock(spec=Tracer)
            mock_get_provider.return_value = mock_provider
            mock_get_tracer.return_value = mock_tracer
            
            # Mock the gapic methods properly
            mock_method = Mock()
            mock_method.__module__ = "google.api_core.gapic_v1.method"
            mock_method.__name__ = "wrap_method"
            
            mock_method_async = Mock()
            mock_method_async.__module__ = "google.api_core.gapic_v1.method_async"
            mock_method_async.__name__ = "wrap_method"
            
            mock_gapic.method.wrap_method = mock_method
            mock_gapic.method_async.wrap_method = mock_method_async
            
            instrumentor = VertexAIInstrumentor()
            instrumentor._instrument()
            
            mock_get_provider.assert_called_once()
            mock_get_tracer.assert_called_once()
            # Should wrap both sync and async gapic methods + Protect.protect
            assert mock_wrap.call_count >= 2

    def test_instrument_with_custom_tracer_provider(self):
        """Test instrumentation with custom tracer provider."""
        with patch('traceai_vertexai.get_tracer') as mock_get_tracer, \
             patch('traceai_vertexai.wrap_function_wrapper') as mock_wrap:
            
            mock_provider = Mock()
            mock_tracer = Mock(spec=Tracer)
            mock_get_tracer.return_value = mock_tracer
            
            instrumentor = VertexAIInstrumentor()
            instrumentor._instrument(tracer_provider=mock_provider)
            
            mock_get_tracer.assert_called_once()
            assert mock_wrap.call_count >= 2

    def test_instrument_with_custom_config(self):
        """Test instrumentation with custom trace config."""
        with patch('traceai_vertexai.get_tracer_provider') as mock_get_provider, \
             patch('traceai_vertexai.get_tracer') as mock_get_tracer, \
             patch('traceai_vertexai.wrap_function_wrapper') as mock_wrap:
            
            mock_provider = Mock()
            mock_tracer = Mock(spec=Tracer)
            mock_get_provider.return_value = mock_provider
            mock_get_tracer.return_value = mock_tracer
            
            custom_config = TraceConfig()
            
            instrumentor = VertexAIInstrumentor()
            instrumentor._instrument(config=custom_config)
            
            assert mock_wrap.call_count >= 2

    def test_instrument_with_invalid_config_type(self):
        """Test instrumentation fails with invalid config type."""
        instrumentor = VertexAIInstrumentor()
        
        with pytest.raises(AssertionError):
            instrumentor._instrument(config="invalid_config")

    def test_uninstrument(self):
        """Test uninstrumentation process."""
        with patch('google.api_core.gapic_v1.method') as mock_method, \
             patch('google.api_core.gapic_v1.method_async') as mock_method_async:
            
            # Mock wrapped methods with proper __wrapped__ attribute
            original_method = Mock()
            original_async_method = Mock()
            
            mock_method.wrap_method = Mock()
            mock_method.wrap_method.__wrapped__ = original_method
            mock_method_async.wrap_method = Mock()
            mock_method_async.wrap_method.__wrapped__ = original_async_method
            
            instrumentor = VertexAIInstrumentor()
            instrumentor._uninstrument()
            
            # Should restore original methods
            assert mock_method.wrap_method == original_method
            assert mock_method_async.wrap_method == original_async_method

    def test_uninstrument_without_wrapped(self):
        """Test uninstrumentation when no wrapped methods exist."""
        with patch('google.api_core.gapic_v1.method') as mock_method, \
             patch('google.api_core.gapic_v1.method_async') as mock_method_async:
            
            # Mock methods without __wrapped__ attribute
            mock_method.wrap_method = Mock()
            mock_method_async.wrap_method = Mock()
            
            instrumentor = VertexAIInstrumentor()
            
            # Should not raise errors
            instrumentor._uninstrument()


class TestWrapper:
    """Test suite for _Wrapper class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_tracer = Mock(spec=Tracer)
        self.wrapper = _Wrapper(tracer=self.mock_tracer)

    def test_initialization(self):
        """Test wrapper initialization."""
        assert self.wrapper._tracer == self.mock_tracer

    def test_call_with_no_request(self):
        """Test wrapper when no prediction service request is found."""
        def mock_function():
            return "response"
        
        # Mock _get_prediction_service_request to return None
        with patch('traceai_vertexai._wrapper._get_prediction_service_request', return_value=None):
            # Apply wrapper as decorator and call
            wrapped_function = self.wrapper(mock_function)
            result = wrapped_function()
            
            assert result == "response"

    def test_call_with_suppressed_instrumentation(self):
        """Test wrapper when instrumentation is suppressed."""
        def mock_function():
            return "response"
        
        with patch('traceai_vertexai._wrapper._get_prediction_service_request') as mock_get_request, \
             patch('traceai_vertexai._wrapper.context_api.get_value', return_value=True):
            
            mock_request = Mock(spec=proto.Message)
            mock_get_request.return_value = mock_request
            
            wrapped_function = self.wrapper(mock_function)
            result = wrapped_function()
            
            assert result == "response"

    def test_call_when_not_instrumented(self):
        """Test wrapper when instrumentation status is False."""
        def mock_function():
            return "response"
        
        with patch('traceai_vertexai._wrapper._get_prediction_service_request') as mock_get_request, \
             patch('traceai_vertexai._wrapper.context_api.get_value', return_value=False):
            
            mock_request = Mock(spec=proto.Message)
            mock_get_request.return_value = mock_request
            
            # Set instrumentation status to False
            self.wrapper._status._IS_INSTRUMENTED = False
            
            wrapped_function = self.wrapper(mock_function)
            result = wrapped_function()
            
            assert result == "response"

    def test_call_successful_with_proto_request(self):
        """Test successful wrapper call with proto.Message request."""
        def mock_function(mock_request):
            return "response"
        
        # Create mock proto message
        mock_request = Mock(spec=proto.Message)
        mock_request.__class__.to_dict = Mock(return_value={"contents": []})
        
        with patch('traceai_vertexai._wrapper._get_prediction_service_request', return_value=mock_request), \
             patch('traceai_vertexai._wrapper.context_api.get_value', return_value=False), \
             patch('traceai_vertexai._wrapper._extract_image_data', return_value={}), \
             patch('traceai_vertexai._wrapper._update_span'), \
             patch('traceai_vertexai._wrapper._finish'), \
             patch('traceai_vertexai._wrapper.use_span'):
            
            self.wrapper._status._IS_INSTRUMENTED = True
            mock_span = Mock()
            self.wrapper._tracer.start_span = Mock(return_value=mock_span)
            
            wrapped_function = self.wrapper(mock_function)
            result = wrapped_function(mock_request)
            
            assert result == "response"
            mock_span.set_attribute.assert_called()

    def test_call_with_exception(self):
        """Test wrapper when wrapped function raises exception."""
        def mock_function(mock_request):
            raise Exception("Test error")
        
        mock_request = Mock(spec=proto.Message)
        mock_request.__class__.to_dict = Mock(return_value={"contents": []})
        
        with patch('traceai_vertexai._wrapper._get_prediction_service_request', return_value=mock_request), \
             patch('traceai_vertexai._wrapper.context_api.get_value', return_value=False), \
             patch('traceai_vertexai._wrapper._extract_image_data', return_value={}), \
             patch('traceai_vertexai._wrapper._update_span'), \
             patch('traceai_vertexai._wrapper.use_span'):
            
            self.wrapper._status._IS_INSTRUMENTED = True
            mock_span = Mock()
            self.wrapper._tracer.start_span = Mock(return_value=mock_span)
            
            wrapped_function = self.wrapper(mock_function)
            with pytest.raises(Exception, match="Test error"):
                wrapped_function(mock_request)
            
            mock_span.record_exception.assert_called()
            mock_span.set_status.assert_called()
            mock_span.end.assert_called()

    def test_call_with_awaitable_result(self):
        """Test wrapper when result is awaitable."""
        import asyncio
        
        # Create a proper coroutine function
        async def async_result():
            return "async_response"
        
        def mock_function(mock_request):
            return async_result()  # Return coroutine
        
        mock_request = Mock(spec=proto.Message)
        mock_request.__class__.to_dict = Mock(return_value={"contents": []})
        
        with patch('traceai_vertexai._wrapper._get_prediction_service_request', return_value=mock_request), \
             patch('traceai_vertexai._wrapper.context_api.get_value', return_value=False), \
             patch('traceai_vertexai._wrapper._extract_image_data', return_value={}), \
             patch('traceai_vertexai._wrapper._update_span'), \
             patch('traceai_vertexai._wrapper._proxy') as mock_proxy, \
             patch('traceai_vertexai._wrapper.use_span'):
            
            self.wrapper._status._IS_INSTRUMENTED = True
            mock_span = Mock()
            self.wrapper._tracer.start_span = Mock(return_value=mock_span)
            
            # Mock proxy to return the coroutine
            mock_proxy.return_value = async_result()
            
            wrapped_function = self.wrapper(mock_function)
            result = wrapped_function(mock_request)
            
            # Should call _proxy for awaitable result
            mock_proxy.assert_called()


class TestCallbackClasses:
    """Test suite for callback classes."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_request = Mock(spec=proto.Message)
        self.mock_span = Mock()

    def test_callback_for_awaitable_initialization(self):
        """Test CallbackForAwaitable initialization."""
        callback = _CallbackForAwaitable(self.mock_request, self.mock_span)
        assert callback._request == self.mock_request

"""
Test suite for Groq framework instrumentation.

Tests the instrumentation of Groq components:
- Completions.create (LLM spans)
- AsyncCompletions.create (async LLM spans)
- Protect.protect (guardrail protection)
- Streaming and non-streaming response handling
- Request/response attribute extraction
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import AsyncGenerator
import json

from opentelemetry import trace as trace_api
from opentelemetry.trace.status import StatusCode
from opentelemetry import context as context_api

from traceai_groq import GroqInstrumentor
from traceai_groq._wrappers import (
    _CompletionsWrapper,
    _AsyncCompletionsWrapper,
    _WithTracer,
    _parse_args,
    _flatten,
)
from traceai_groq._utils import (
    _io_value_and_type,
    _as_input_attributes,
    _as_output_attributes,
    _finish_tracing,
    _extract_eval_input,
    _process_response,
    _as_raw_output,
    _as_streaming_output,
    _to_dict,
)
from traceai_groq._request_attributes_extractor import _RequestAttributesExtractor
from traceai_groq._response_attributes_extractor import _ResponseAttributesExtractor
from traceai_groq._with_span import _WithSpan


class TestGroqInstrumentor:
    """Test the main Groq instrumentor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.instrumentor = GroqInstrumentor()

    def test_instrumentation_dependencies(self):
        """Test instrumentation dependencies."""
        deps = self.instrumentor.instrumentation_dependencies()
        assert ("groq >= 0.9.0") == deps

    @patch("traceai_groq.wrap_function_wrapper")
    @patch("traceai_groq.trace_api.get_tracer")
    def test_instrument_basic(self, mock_get_tracer, mock_wrap_function):
        """Test basic instrumentation setup."""
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer
        
        # Mock the Protect import to avoid ImportError
        with patch("traceai_groq.Protect") as mock_protect:
            mock_protect.protect = Mock()
            
            self.instrumentor._instrument()
        
        # Verify tracer setup
        mock_get_tracer.assert_called_once()
        
        # Verify function wrapping - should wrap completions and protect
        assert mock_wrap_function.call_count >= 2
        
        # Check that originals are stored
        assert hasattr(self.instrumentor, '_original_completions_create')
        assert hasattr(self.instrumentor, '_original_async_completions_create')

    def test_uninstrument(self):
        """Test uninstrumentation."""
        # Set up mock originals
        self.instrumentor._original_completions_create = Mock()
        self.instrumentor._original_async_completions_create = Mock()
        
        with patch("traceai_groq.import_module") as mock_import:
            mock_module = Mock()
            mock_module.Completions = Mock()
            mock_module.AsyncCompletions = Mock()
            mock_import.return_value = mock_module
            
            self.instrumentor._uninstrument()
            
            # Verify originals are restored
            assert mock_module.Completions.create == self.instrumentor._original_completions_create
            assert mock_module.AsyncCompletions.create == self.instrumentor._original_async_completions_create


class TestCompletionsWrapper:
    """Test the sync completions wrapper."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_tracer = MagicMock(spec=trace_api.Tracer)
        self.mock_span = MagicMock()
        self.wrapper = _CompletionsWrapper(tracer=self.mock_tracer)

    def test_completions_wrapper_basic(self):
        """Test basic completions wrapper operation."""
        # Mock request parameters
        request_params = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "llama-3.1-70b-versatile",
            "stream": False
        }
        
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Hello! How can I help?"
        
        def wrapped_func(*args, **kwargs):
            return mock_response
        
        # Mock the span context manager
        mock_span_context = MagicMock()
        mock_with_span = Mock()
        mock_span_context.__enter__.return_value = mock_with_span
        mock_span_context.__exit__.return_value = None
        self.wrapper._start_as_current_span = Mock(return_value=mock_span_context)
        
        # Test the wrapper
        result = self.wrapper(wrapped_func, None, (), request_params)
        
        assert result == mock_response
        self.wrapper._start_as_current_span.assert_called_once()

    def test_completions_wrapper_streaming(self):
        """Test streaming completions wrapper operation."""
        # Mock request parameters
        request_params = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "llama-3.1-70b-versatile",
            "stream": True
        }
        
        # Mock streaming response
        mock_chunk1 = Mock()
        mock_chunk1.choices = [{"delta": {"content": "Hello"}}]
        mock_chunk2 = Mock()
        mock_chunk2.choices = [{"delta": {"content": " there!"}}]
        
        def mock_streaming_response():
            yield mock_chunk1
            yield mock_chunk2
        
        def wrapped_func(*args, **kwargs):
            return mock_streaming_response()
        
        # Mock the span context manager
        mock_span_context = MagicMock()
        mock_with_span = Mock()
        mock_span_context.__enter__.return_value = mock_with_span
        mock_span_context.__exit__.return_value = None
        self.wrapper._start_as_current_span = Mock(return_value=mock_span_context)
        
        # Test the wrapper
        result = self.wrapper(wrapped_func, None, (), request_params)
        
        # Consume the generator
        chunks = list(result)
        assert len(chunks) == 2

    def test_completions_wrapper_exception(self):
        """Test completions wrapper exception handling."""
        def wrapped_func(*args, **kwargs):
            raise ValueError("Test error")
        
        # Mock the span context manager
        mock_span_context = MagicMock()
        mock_with_span = Mock()
        mock_span_context.__enter__.return_value = mock_with_span
        mock_span_context.__exit__.return_value = None
        self.wrapper._start_as_current_span = Mock(return_value=mock_span_context)
        
        # Test exception handling
        with pytest.raises(ValueError, match="Test error"):
            self.wrapper(wrapped_func, None, (), {})
        
        # Verify exception was recorded
        mock_with_span.record_exception.assert_called_once()

    def test_completions_wrapper_suppressed(self):
        """Test completions wrapper when instrumentation is suppressed."""
        with patch("traceai_groq._wrappers.context_api.get_value") as mock_get_value:
            mock_get_value.return_value = True
            
            mock_response = Mock()
            def wrapped_func(*args, **kwargs):
                return mock_response
            
            result = self.wrapper(wrapped_func, None, (), {})
            
            assert result == mock_response


class TestAsyncCompletionsWrapper:
    """Test the async completions wrapper."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_tracer = MagicMock(spec=trace_api.Tracer)
        self.wrapper = _AsyncCompletionsWrapper(tracer=self.mock_tracer)

    @pytest.mark.asyncio
    async def test_async_completions_wrapper_basic(self):
        """Test basic async completions wrapper operation."""
        # Mock request parameters
        request_params = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "llama-3.1-70b-versatile",
            "stream": False
        }
        
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Hello! How can I help?"
        
        async def wrapped_func(*args, **kwargs):
            return mock_response
        
        # Mock the span context manager
        mock_span_context = MagicMock()
        mock_with_span = Mock()
        mock_span_context.__enter__.return_value = mock_with_span
        mock_span_context.__exit__.return_value = None
        self.wrapper._start_as_current_span = Mock(return_value=mock_span_context)
        
        # Test the wrapper
        result = await self.wrapper(wrapped_func, None, (), request_params)
        
        assert result == mock_response
        self.wrapper._start_as_current_span.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_completions_wrapper_streaming(self):
        """Test async streaming completions wrapper operation."""
        # Mock request parameters
        request_params = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "llama-3.1-70b-versatile",
            "stream": True
        }
        
        # Mock streaming response
        async def mock_streaming_response():
            yield {"choices": [{"delta": {"content": "Hello"}}]}
            yield {"choices": [{"delta": {"content": " there!"}}]}

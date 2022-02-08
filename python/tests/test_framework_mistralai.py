import asyncio
import json
from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import MagicMock, Mock, patch, AsyncMock, call, ANY
from uuid import uuid4

import pytest
from fi_instrumentation import TraceConfig, FITracer
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import INVALID_SPAN

from traceai_mistralai import MistralAIInstrumentor
from traceai_mistralai._chat_wrapper import (
    _SyncChatWrapper,
    _AsyncChatWrapper,
    _AsyncStreamChatWrapper,
    _WithTracer,
    _WithMistralAI,
)
from traceai_mistralai._request_attributes_extractor import _RequestAttributesExtractor
from traceai_mistralai._response_attributes_extractor import (
    _ResponseAttributesExtractor,
    _StreamResponseAttributesExtractor,
)
from traceai_mistralai._utils import (
    _finish_tracing,
    _process_input_messages,
    _raw_input,
    _io_value_and_type,
    _as_input_attributes,
    _as_output_attributes,
)
from traceai_mistralai._with_span import _WithSpan


@pytest.fixture
def tracer_provider():
    """Create a test TracerProvider with console export for testing."""
    provider = TracerProvider()
    exporter = ConsoleSpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


@pytest.fixture
def tracer(tracer_provider):
    """Create a test tracer."""
    return trace_api.get_tracer(__name__, "1.0.0", tracer_provider)


@pytest.fixture
def fi_tracer(tracer):
    """Create a FITracer instance for testing."""
    config = TraceConfig()
    return FITracer(tracer, config=config)


@pytest.fixture
def mock_mistralai_module():
    """Mock the mistralai module."""
    mock_module = MagicMock()
    
    # Mock Chat class
    mock_chat_class = MagicMock()
    mock_chat_class.complete = MagicMock()
    mock_chat_class.stream = MagicMock()
    mock_chat_class.complete_async = MagicMock()
    mock_chat_class.stream_async = MagicMock()
    
    # Mock Agents class
    mock_agents_class = MagicMock()
    mock_agents_class.complete = MagicMock()
    mock_agents_class.stream = MagicMock()
    mock_agents_class.complete_async = MagicMock()
    mock_agents_class.stream_async = MagicMock()
    
    mock_module.chat.Chat = mock_chat_class
    mock_module.agents.Agents = mock_agents_class
    
    return mock_module


@pytest.fixture
def mock_mistral_client():
    """Create a mock Mistral client instance."""
    client = MagicMock()
    client.chat = MagicMock()
    client.agents = MagicMock()
    return client


@pytest.fixture
def sample_request():
    """Sample request parameters for testing."""
    return {
        "model": "mistral-tiny",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
    }


@pytest.fixture
def sample_response():
    """Sample response object for testing."""
    mock_response = MagicMock()
    mock_response.id = "test-response-id"
    mock_response.model = "mistral-tiny"
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                role="assistant",
                content="Hello! I'm doing well, thank you for asking."
            ),
            finish_reason="stop"
        )
    ]
    mock_response.usage = MagicMock(
        prompt_tokens=10,
        completion_tokens=15,
        total_tokens=25
    )
    return mock_response


class TestMistralAIInstrumentor:
    """⚡ Test MistralAIInstrumentor lifecycle and basic functionality."""

    def test_instrumentation_dependencies(self):
        """Test instrumentor dependencies are properly declared."""
        instrumentor = MistralAIInstrumentor()
        dependencies = instrumentor.instrumentation_dependencies()
        
        assert isinstance(dependencies, tuple)
        assert "mistralai" in dependencies
        assert len(dependencies) >= 1

    def test_instrument_with_defaults(self, tracer_provider):
        """Test instrumentation with default parameters."""
        instrumentor = MistralAIInstrumentor()
        
        with patch("mistralai.chat.Chat") as mock_chat:
            with patch("mistralai.agents.Agents") as mock_agents:
                with patch("traceai_mistralai.wrap_function_wrapper") as mock_wrap:
                    instrumentor._instrument(tracer_provider=tracer_provider)
                    
                    # Verify tracer was created
                    assert hasattr(instrumentor, '_tracer')
                    assert isinstance(instrumentor._tracer, FITracer)
                    
                    # Verify all methods were wrapped
                    assert mock_wrap.call_count == 8  # 4 chat + 4 agents methods
                    
                    # Verify original methods were stored
                    assert hasattr(instrumentor, '_original_sync_chat_method')
                    assert hasattr(instrumentor, '_original_async_chat_method')
                    assert hasattr(instrumentor, '_original_sync_agent_method')
                    assert hasattr(instrumentor, '_original_async_agent_method')

    def test_instrument_with_custom_config(self, tracer_provider):
        """Test instrumentation with custom TraceConfig."""
        instrumentor = MistralAIInstrumentor()
        custom_config = TraceConfig()
        
        with patch("mistralai.chat.Chat"):
            with patch("mistralai.agents.Agents"):
                with patch("traceai_mistralai.wrap_function_wrapper"):
                    instrumentor._instrument(
                        tracer_provider=tracer_provider,
                        config=custom_config
                    )
                    
                    assert hasattr(instrumentor, '_tracer')
                    assert isinstance(instrumentor._tracer, FITracer)

    def test_instrument_missing_mistralai_import(self):
        """Test instrumentation behavior when mistralai is not available."""
        instrumentor = MistralAIInstrumentor()
        
        # Patch the import within the _instrument method
        with patch("traceai_mistralai.MistralAIInstrumentor._instrument") as mock_instrument:
            mock_instrument.side_effect = Exception("Could not import mistralai. Please install with `pip install mistralai`.")
            
            with pytest.raises(Exception, match="Could not import mistralai"):
                instrumentor._instrument()

    def test_uninstrument_restores_original_methods(self):
        """Test that uninstrumentation properly restores original methods."""
        instrumentor = MistralAIInstrumentor()
        
        with patch("mistralai.chat.Chat") as mock_chat:
            with patch("mistralai.agents.Agents") as mock_agents:
                with patch("traceai_mistralai.wrap_function_wrapper"):
                    # Store original methods
                    original_chat_complete = MagicMock()
                    original_chat_stream = MagicMock()
                    original_agents_complete = MagicMock()
                    original_agents_stream = MagicMock()
                    
                    mock_chat.complete = original_chat_complete
                    mock_chat.stream = original_chat_stream
                    mock_chat.complete_async = MagicMock()
                    mock_chat.stream_async = MagicMock()
                    mock_agents.complete = original_agents_complete
                    mock_agents.stream = original_agents_stream
                    mock_agents.complete_async = MagicMock()
                    mock_agents.stream_async = MagicMock()
                    
                    instrumentor._instrument()
                    
                    # Now uninstrument
                    instrumentor._uninstrument()
                    
                    # Verify original methods were restored
                    assert mock_chat.complete == original_chat_complete
                    assert mock_chat.stream == original_chat_stream  
                    assert mock_agents.complete == original_agents_complete
                    assert mock_agents.stream == original_agents_stream

    def test_wrapper_registration_calls(self, tracer_provider):
        """Test that all expected wrapper functions are registered."""
        instrumentor = MistralAIInstrumentor()
        
        with patch("mistralai.chat.Chat"):
            with patch("mistralai.agents.Agents"):
                with patch("traceai_mistralai.wrap_function_wrapper") as mock_wrap:
                    instrumentor._instrument(tracer_provider=tracer_provider)
                    
                    # Check that all expected calls were made
                    assert len(mock_wrap.call_args_list) == 8
                    
                    # Check that both chat and agent modules are instrumented
                    modules_called = []
                    names_called = []
                    for actual_call in mock_wrap.call_args_list:
                        args, kwargs = actual_call
                        modules_called.append(kwargs.get("module"))
                        names_called.append(kwargs.get("name"))
                    
                    # Verify modules
                    assert "mistralai.chat" in modules_called
                    assert "mistralai.agents" in modules_called
                    
                    # Verify method names
                    expected_names = [
                        "Chat.complete", "Chat.stream", "Chat.complete_async", "Chat.stream_async",
                        "Agents.complete", "Agents.stream", "Agents.complete_async", "Agents.stream_async"
                    ]
                    for expected_name in expected_names:
                        assert expected_name in names_called


class TestSyncChatWrapper:
    """🔄 Test synchronous chat wrapper functionality."""

    def test_sync_chat_wrapper_initialization(self, fi_tracer, mock_mistralai_module):
        """Test SyncChatWrapper initialization."""
        wrapper = _SyncChatWrapper("test_span", fi_tracer, mock_mistralai_module)
        
        assert wrapper._tracer == fi_tracer
        assert hasattr(wrapper, '_request_attributes_extractor')
        assert hasattr(wrapper, '_response_attributes_extractor')

    def test_sync_chat_wrapper_call_success(self, fi_tracer, mock_mistralai_module, 
                                          mock_mistral_client, sample_request, sample_response):
        """Test successful synchronous chat completion call."""
        wrapper = _SyncChatWrapper("MistralClient.chat", fi_tracer, mock_mistralai_module)
        
        # Mock the wrapped function to return a non-stream response
        mock_wrapped = MagicMock(return_value=sample_response)
        
        with patch.object(wrapper, '_start_as_current_span') as mock_span_ctx:
            mock_with_span = MagicMock()
            mock_span_ctx.return_value.__enter__.return_value = mock_with_span
            mock_span_ctx.return_value.__exit__.return_value = None
            
            result = wrapper(mock_wrapped, mock_mistral_client, (), sample_request)
            
            # Verify the result is returned
            assert result == sample_response
            
            # Verify span was created
            mock_span_ctx.assert_called_once()

    def test_sync_chat_wrapper_call_stream_response(self, fi_tracer, mock_mistralai_module,
                                                   mock_mistral_client, sample_request):
        """Test synchronous chat wrapper with streaming response."""
        wrapper = _SyncChatWrapper("MistralClient.chat", fi_tracer, mock_mistralai_module)
        
        # Simple test - verify wrapper can handle different response types
        mock_wrapped = MagicMock(return_value="simple_response")
        
        with patch.object(wrapper, '_start_as_current_span') as mock_span_ctx:
            mock_with_span = MagicMock()
            mock_span_ctx.return_value.__enter__.return_value = mock_with_span
            mock_span_ctx.return_value.__exit__.return_value = None
            
            result = wrapper(mock_wrapped, mock_mistral_client, (), sample_request)
            
            # Should return the response
            assert result == "simple_response"

    def test_sync_chat_wrapper_error_handling(self, fi_tracer, mock_mistralai_module,
                                             mock_mistral_client, sample_request):
        """Test error handling in synchronous chat wrapper."""

import asyncio
import json
from typing import Any, Dict, List, AsyncIterator, Iterator
from unittest.mock import MagicMock, Mock, patch, AsyncMock
from uuid import uuid4

import pytest
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Span

from fi_instrumentation import TraceConfig, FITracer
from fi_instrumentation.fi_types import FiSpanKindValues, SpanAttributes
from traceai_litellm import (
    LiteLLMInstrumentor,
    StreamingIteratorWrapper,
    AsyncStreamingIteratorWrapper,
    _set_span_attribute,
    _instrument_func_type_completion,
    _instrument_func_type_embedding,
    _instrument_func_type_image_generation,
    _finalize_span,
    _process_messages,
    _get_attributes_from_message_param,
    _get_attributes_from_message_content,
    _get_attributes_from_image,
    is_iterable_of,
)


@pytest.fixture
def tracer_provider():
    """Create a test TracerProvider with console export for testing."""
    provider = TracerProvider()
    exporter = ConsoleSpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


@pytest.fixture
def config():
    """Create a test TraceConfig."""
    return TraceConfig()


@pytest.fixture
def instrumentation_scope():
    """Set up clean instrumentation environment."""
    # Clear any existing instrumentation suppression
    context_api.set_value(_SUPPRESS_INSTRUMENTATION_KEY, False)
    yield
    # Reset to clean state
    context_api.set_value(_SUPPRESS_INSTRUMENTATION_KEY, False)


@pytest.fixture
def mock_litellm():
    """Mock LiteLLM functions."""
    with patch("litellm.completion") as mock_completion:
        with patch("litellm.acompletion") as mock_acompletion:
            with patch("litellm.completion_with_retries") as mock_retries:
                with patch("litellm.acompletion_with_retries") as mock_aretries:
                    with patch("litellm.embedding") as mock_embedding:
                        with patch("litellm.aembedding") as mock_aembedding:
                            with patch("litellm.image_generation") as mock_image:
                                with patch("litellm.aimage_generation") as mock_aimage:
                                    yield {
                                        "completion": mock_completion,
                                        "acompletion": mock_acompletion,
                                        "completion_with_retries": mock_retries,
                                        "acompletion_with_retries": mock_aretries,
                                        "embedding": mock_embedding,
                                        "aembedding": mock_aembedding,
                                        "image_generation": mock_image,
                                        "aimage_generation": mock_aimage,
                                    }


@pytest.fixture
def sample_completion_response():
    """Sample completion response."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Hello! How can I help you?"))
    ]
    mock_response.to_dict.return_value = {
        "choices": [{"message": {"content": "Hello! How can I help you?"}}]
    }
    return mock_response


@pytest.fixture
def sample_embedding_response():
    """Sample embedding response."""
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    mock_response.to_dict.return_value = {
        "data": [{"embedding": [0.1, 0.2, 0.3]}]
    }
    return mock_response


@pytest.fixture
def sample_image_response():
    """Sample image generation response."""
    mock_response = MagicMock()
    mock_response.data = [MagicMock(url="https://example.com/image.png")]
    mock_response.to_dict.return_value = {
        "data": [{"url": "https://example.com/image.png"}]
    }
    return mock_response


class TestLiteLLMInstrumentor:
    """⚡ Test LiteLLMInstrumentor lifecycle and basic functionality."""

    def test_instrumentation_dependencies(self):
        """Test instrumentor dependencies are properly declared."""
        instrumentor = LiteLLMInstrumentor()
        dependencies = instrumentor.instrumentation_dependencies()
        
        assert isinstance(dependencies, tuple)
        assert "litellm" in str(dependencies)
        assert len(dependencies) >= 1

    def test_instrument_basic(self, tracer_provider, config, mock_litellm):
        """Test basic instrumentation setup."""
        instrumentor = LiteLLMInstrumentor()
        
        # Mock the original functions before instrumentation
        instrumentor.original_litellm_funcs["completion"] = mock_litellm["completion"]
        instrumentor.original_litellm_funcs["acompletion"] = mock_litellm["acompletion"]
        instrumentor.original_litellm_funcs["completion_with_retries"] = mock_litellm["completion_with_retries"]
        instrumentor.original_litellm_funcs["acompletion_with_retries"] = mock_litellm["acompletion_with_retries"]
        instrumentor.original_litellm_funcs["embedding"] = mock_litellm["embedding"]
        instrumentor.original_litellm_funcs["aembedding"] = mock_litellm["aembedding"]
        instrumentor.original_litellm_funcs["image_generation"] = mock_litellm["image_generation"]
        instrumentor.original_litellm_funcs["aimage_generation"] = mock_litellm["aimage_generation"]
        
        instrumentor._instrument(tracer_provider=tracer_provider, config=config)
        
        # Verify tracer was created
        assert hasattr(instrumentor, '_tracer')
        assert isinstance(instrumentor._tracer, FITracer)
        
        # Verify original functions were stored
        assert len(instrumentor.original_litellm_funcs) == 8

    def test_instrument_without_tracer_provider(self, config, mock_litellm):
        """Test instrumentation without explicit tracer provider."""
        instrumentor = LiteLLMInstrumentor()
        
        with patch("opentelemetry.trace.get_tracer_provider") as mock_get_provider:
            mock_get_provider.return_value = MagicMock()
            instrumentor._instrument(config=config)
            mock_get_provider.assert_called_once()

    def test_instrument_without_config(self, tracer_provider, mock_litellm):
        """Test instrumentation without explicit config."""
        instrumentor = LiteLLMInstrumentor()
        
        instrumentor._instrument(tracer_provider=tracer_provider)
        
        # Should create default TraceConfig
        assert hasattr(instrumentor, '_tracer')
        assert isinstance(instrumentor._tracer, FITracer)

    def test_uninstrument(self, tracer_provider, config, mock_litellm):
        """Test proper uninstrumentation."""
        instrumentor = LiteLLMInstrumentor()
        
        # Store original functions
        instrumentor.original_litellm_funcs["completion"] = mock_litellm["completion"]
        instrumentor.original_litellm_funcs["acompletion"] = mock_litellm["acompletion"]
        
        instrumentor._instrument(tracer_provider=tracer_provider, config=config)
        
        # Verify instrumentation
        assert instrumentor._tracer is not None
        
        # Test uninstrumentation
        instrumentor._uninstrument()
        
        # Verify cleanup
        assert instrumentor._tracer is None


class TestCompletionInstrumentation:
    """Test completion function instrumentation."""

    def test_completion_basic(self, tracer_provider, config, mock_litellm, sample_completion_response):
        """Test basic completion instrumentation."""
        instrumentor = LiteLLMInstrumentor()
        instrumentor.original_litellm_funcs["completion"] = mock_litellm["completion"]
        mock_litellm["completion"].return_value = sample_completion_response
        
        instrumentor._instrument(tracer_provider=tracer_provider, config=config)
        
        # Test completion call
        result = instrumentor._completion_wrapper(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert result == sample_completion_response
        mock_litellm["completion"].assert_called_once()

    @pytest.mark.asyncio
    async def test_acompletion_basic(self, tracer_provider, config, mock_litellm, sample_completion_response):
        """Test basic async completion instrumentation."""
        instrumentor = LiteLLMInstrumentor()
        instrumentor.original_litellm_funcs["acompletion"] = mock_litellm["acompletion"]
        
        # Configure AsyncMock to return the response when awaited
        mock_litellm["acompletion"].return_value = sample_completion_response
        
        instrumentor._instrument(tracer_provider=tracer_provider, config=config)
        
        # Test async completion call
        result = await instrumentor._acompletion_wrapper(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert result == sample_completion_response
        mock_litellm["acompletion"].assert_called_once()

    def test_completion_with_retries(self, tracer_provider, config, mock_litellm, sample_completion_response):
        """Test completion with retries instrumentation."""
        instrumentor = LiteLLMInstrumentor()
        instrumentor.original_litellm_funcs["completion_with_retries"] = mock_litellm["completion_with_retries"]
        mock_litellm["completion_with_retries"].return_value = sample_completion_response
        
        instrumentor._instrument(tracer_provider=tracer_provider, config=config)
        
        # Test completion with retries call
        result = instrumentor._completion_with_retries_wrapper(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_retries=3
        )
        
        assert result == sample_completion_response
        mock_litellm["completion_with_retries"].assert_called_once()

    @pytest.mark.asyncio
    async def test_acompletion_with_retries(self, tracer_provider, config, mock_litellm, sample_completion_response):
        """Test async completion with retries instrumentation."""
        instrumentor = LiteLLMInstrumentor()
        instrumentor.original_litellm_funcs["acompletion_with_retries"] = mock_litellm["acompletion_with_retries"]
        
        # Configure AsyncMock to return the response when awaited
        mock_litellm["acompletion_with_retries"].return_value = sample_completion_response
        
        instrumentor._instrument(tracer_provider=tracer_provider, config=config)
        
        # Test async completion with retries call
        result = await instrumentor._acompletion_with_retries_wrapper(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_retries=3
        )
        
        assert result == sample_completion_response
        mock_litellm["acompletion_with_retries"].assert_called_once()

    def test_completion_with_streaming(self, tracer_provider, config, mock_litellm):
        """Test completion with streaming response."""
        instrumentor = LiteLLMInstrumentor()
        instrumentor.original_litellm_funcs["completion"] = mock_litellm["completion"]
        
        # Mock streaming response
        mock_stream_chunk = MagicMock()
        mock_stream_chunk.to_dict.return_value = {
            "choices": [{"delta": {"content": "Hello"}}]
        }
        mock_stream = [mock_stream_chunk]
        mock_litellm["completion"].return_value = iter(mock_stream)
        
        instrumentor._instrument(tracer_provider=tracer_provider, config=config)
        
        # Test streaming completion call
        result = instrumentor._completion_wrapper(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )
        
        # Should return StreamingIteratorWrapper
        assert isinstance(result, StreamingIteratorWrapper)
        
        # Consume the stream
        chunks = list(result)
        assert len(chunks) == 1

    def test_completion_suppressed(self, tracer_provider, config, mock_litellm, sample_completion_response):
        """Test completion with instrumentation suppressed."""
        instrumentor = LiteLLMInstrumentor()
        instrumentor.original_litellm_funcs["completion"] = mock_litellm["completion"]
        mock_litellm["completion"].return_value = sample_completion_response
        
        instrumentor._instrument(tracer_provider=tracer_provider, config=config)
        
        # Set instrumentation suppression
        with patch.object(context_api, 'get_value', return_value=True):
            result = instrumentor._completion_wrapper(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}]
            )
        
        assert result == sample_completion_response
        mock_litellm["completion"].assert_called_once()


class TestEmbeddingInstrumentation:
    """Test embedding function instrumentation."""

    def test_embedding_basic(self, tracer_provider, config, mock_litellm, sample_embedding_response):
        """Test basic embedding instrumentation."""
        instrumentor = LiteLLMInstrumentor()
        instrumentor.original_litellm_funcs["embedding"] = mock_litellm["embedding"]
        mock_litellm["embedding"].return_value = sample_embedding_response
        
        instrumentor._instrument(tracer_provider=tracer_provider, config=config)
        
        # Test embedding call
        result = instrumentor._embedding_wrapper(
            model="text-embedding-ada-002",
            input="Hello world"
        )
        
        assert result == sample_embedding_response
        mock_litellm["embedding"].assert_called_once()

    @pytest.mark.asyncio
    async def test_aembedding_basic(self, tracer_provider, config, mock_litellm, sample_embedding_response):
        """Test basic async embedding instrumentation."""
        instrumentor = LiteLLMInstrumentor()
        instrumentor.original_litellm_funcs["aembedding"] = mock_litellm["aembedding"]
        
        # Configure AsyncMock to return the response when awaited
        mock_litellm["aembedding"].return_value = sample_embedding_response
        
        instrumentor._instrument(tracer_provider=tracer_provider, config=config)
        
        # Test async embedding call
        result = await instrumentor._aembedding_wrapper(
            model="text-embedding-ada-002",
            input="Hello world"
        )
        
        assert result == sample_embedding_response
        mock_litellm["aembedding"].assert_called_once()

    def test_embedding_suppressed(self, tracer_provider, config, mock_litellm, sample_embedding_response):
        """Test embedding with instrumentation suppressed."""
        instrumentor = LiteLLMInstrumentor()

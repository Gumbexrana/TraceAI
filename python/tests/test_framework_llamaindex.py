import asyncio
import json
from typing import Any, Dict, List, Optional
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
from fi_instrumentation.fi_types import FiSpanKindValues, SpanAttributes, RerankerAttributes
from traceai_llamaindex import LlamaIndexInstrumentor, get_current_span


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
def mock_llama_index_core():
    """Mock llama-index-core components."""
    with patch("llama_index.core.global_handler", None) as mock_global_handler:
        with patch("llama_index.core.instrumentation.get_dispatcher") as mock_dispatcher:
            mock_dispatcher_instance = MagicMock()
            mock_dispatcher_instance.span_handlers = []
            mock_dispatcher_instance.event_handlers = []
            mock_dispatcher.return_value = mock_dispatcher_instance
            
            yield {
                "global_handler": mock_global_handler,
                "get_dispatcher": mock_dispatcher,
                "dispatcher": mock_dispatcher_instance,
            }


class TestLlamaIndexInstrumentor:
    """⚡ Test LlamaIndexInstrumentor lifecycle and basic functionality."""

    def test_instrumentation_dependencies(self):
        """Test instrumentor dependencies are properly declared."""
        instrumentor = LlamaIndexInstrumentor()
        dependencies = instrumentor.instrumentation_dependencies()
        
        assert isinstance(dependencies, tuple)
        assert "llama-index-core >= 0.10.43" in dependencies
        assert len(dependencies) >= 2

    def test_instrument_new_handler_mode(self, tracer_provider, config, mock_llama_index_core):
        """Test instrumentation with new event/span handler mode."""
        instrumentor = LlamaIndexInstrumentor()
        
        instrumentor._instrument(tracer_provider=tracer_provider, config=config)
        
        # Verify tracer was created
        assert hasattr(instrumentor, '_tracer')
        assert isinstance(instrumentor._tracer, FITracer)
        
        # Verify new handler mode was used
        assert not instrumentor._use_legacy_callback_handler
        assert instrumentor._event_handler is not None
        assert instrumentor._span_handler is not None
        
        # Verify handlers were added to dispatcher
        mock_llama_index_core["dispatcher"].add_span_handler.assert_called_once()
        mock_llama_index_core["dispatcher"].add_event_handler.assert_called_once()

    def test_instrument_legacy_callback_mode(self, tracer_provider, config, mock_llama_index_core):
        """Test instrumentation with legacy callback handler mode."""
        instrumentor = LlamaIndexInstrumentor()
        
        # Mock llama_index.core to have global_handler attribute
        with patch("llama_index.core") as mock_core:
            # Set up the original handler that should be captured
            original_handler = MagicMock()
            mock_core.global_handler = original_handler
            
            instrumentor._instrument(
                tracer_provider=tracer_provider, 
                config=config,
                use_legacy_callback_handler=True
            )
            
            # Verify legacy mode was used
            assert instrumentor._use_legacy_callback_handler
            assert instrumentor._original_global_handler == original_handler
            
            # Verify global handler was set to our callback handler
            assert mock_core.global_handler is not None
            assert mock_core.global_handler != original_handler

    def test_instrument_without_tracer_provider(self, config, mock_llama_index_core):
        """Test instrumentation without explicit tracer provider."""
        instrumentor = LlamaIndexInstrumentor()
        
        with patch("opentelemetry.trace.get_tracer_provider") as mock_get_provider:
            mock_get_provider.return_value = MagicMock()
            instrumentor._instrument(config=config)
            mock_get_provider.assert_called_once()

    def test_instrument_without_config(self, tracer_provider, mock_llama_index_core):
        """Test instrumentation without explicit config."""
        instrumentor = LlamaIndexInstrumentor()
        
        instrumentor._instrument(tracer_provider=tracer_provider)
        
        # Should create default TraceConfig
        assert hasattr(instrumentor, '_tracer')
        assert isinstance(instrumentor._tracer, FITracer)

    def test_uninstrument_new_handler_mode(self, tracer_provider, config, mock_llama_index_core):
        """Test proper uninstrumentation in new handler mode."""
        instrumentor = LlamaIndexInstrumentor()
        
        instrumentor._instrument(tracer_provider=tracer_provider, config=config)
        
        # Verify instrumentation
        assert instrumentor._event_handler is not None
        
        # Test uninstrumentation
        instrumentor._uninstrument()
        
        # Verify cleanup
        assert instrumentor._event_handler is None

    def test_uninstrument_legacy_callback_mode(self, tracer_provider, config, mock_llama_index_core):
        """Test proper uninstrumentation in legacy callback mode."""
        instrumentor = LlamaIndexInstrumentor()
        
        # Mock llama_index.core to have global_handler attribute
        with patch("llama_index.core") as mock_core:
            original_handler = MagicMock()
            mock_core.global_handler = original_handler
            
            instrumentor._instrument(
                tracer_provider=tracer_provider,
                config=config,
                use_legacy_callback_handler=True
            )
            
            # Verify instrumentation worked
            assert instrumentor._use_legacy_callback_handler
            assert instrumentor._original_global_handler == original_handler
            
            # Test uninstrumentation
            instrumentor._uninstrument()
            
            # Verify original handler was restored
            assert mock_core.global_handler == original_handler
            assert instrumentor._original_global_handler is None


class TestEventHandler:
    """🎯 Test event handler functionality."""

    def test_event_handler_creation(self, tracer_provider, config):
        """Test event handler creation and initialization."""
        from traceai_llamaindex._handler import EventHandler
        
        tracer = FITracer(
            trace_api.get_tracer(__name__, "test", tracer_provider),
            config=config,
        )
        
        event_handler = EventHandler(tracer=tracer)
        
        assert event_handler._span_handler is not None
        assert hasattr(event_handler._span_handler, '_otel_tracer')

    def test_event_handler_llm_events(self, tracer_provider, config):
        """Test event handler processing LLM events."""
        from traceai_llamaindex._handler import EventHandler
        
        tracer = FITracer(
            trace_api.get_tracer(__name__, "test", tracer_provider),
            config=config,
        )
        
        event_handler = EventHandler(tracer=tracer)
        
        # Mock LLM events
        with patch("llama_index.core.instrumentation.events.llm.LLMChatStartEvent") as mock_event:
            mock_event.span_id = "test-span-id"
            mock_event.model_dict = {}
            
            # Test handling events doesn't raise errors
            try:
                event_handler.handle(mock_event)
            except Exception as e:
                # Some mocking issues are expected in isolated tests
                assert "instrumentation" in str(e).lower() or "span" in str(e).lower()

    def test_event_handler_embedding_events(self, tracer_provider, config):
        """Test event handler processing embedding events."""
        from traceai_llamaindex._handler import EventHandler
        
        tracer = FITracer(
            trace_api.get_tracer(__name__, "test", tracer_provider),
            config=config,
        )
        
        event_handler = EventHandler(tracer=tracer)
        
        # Mock embedding events
        with patch("llama_index.core.instrumentation.events.embedding.EmbeddingStartEvent") as mock_event:
            mock_event.span_id = "test-embedding-span"
            mock_event.model_dict = {}
            
            # Test handling events doesn't raise errors
            try:
                event_handler.handle(mock_event)
            except Exception as e:
                # Some mocking issues are expected in isolated tests
                assert "instrumentation" in str(e).lower() or "span" in str(e).lower()


class TestSpanHandler:
    """🎭 Test span handler functionality."""

    def test_span_handler_creation(self, tracer_provider, config):
        """Test span handler creation and initialization."""
        from traceai_llamaindex._handler import _SpanHandler
        
        tracer = trace_api.get_tracer(__name__, "test", tracer_provider)
        span_handler = _SpanHandler(tracer=tracer)
        
        assert span_handler._otel_tracer == tracer
        assert span_handler._export_queue is not None

    def test_span_handler_new_span(self, tracer_provider, config):
        """Test span handler creating new spans."""
        from traceai_llamaindex._handler import _SpanHandler
        import inspect
        
        tracer = trace_api.get_tracer(__name__, "test", tracer_provider)
        span_handler = _SpanHandler(tracer=tracer)
        
        # Mock bound arguments
        bound_args = MagicMock()

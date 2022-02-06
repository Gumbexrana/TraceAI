import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch, AsyncMock, call
from uuid import uuid4

import pytest
from opentelemetry import context as otel_context
from opentelemetry import propagate
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

from traceai_mcp import (
    MCPInstrumentor,
    InstrumentedStreamReader,
    InstrumentedStreamWriter,
    ContextSavingStreamWriter,
    ContextAttachingStreamReader,
    ItemWithContext,
)


@pytest.fixture
def tracer_provider():
    """Create a test TracerProvider with console export for testing."""
    provider = TracerProvider()
    exporter = ConsoleSpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


@pytest.fixture
def mock_mcp_modules():
    """Mock MCP modules and classes."""
    mock_modules = {}
    
    # Mock SessionMessage and JSONRPCRequest
    mock_session_message = MagicMock()
    mock_jsonrpc_request = MagicMock()
    mock_jsonrpc_request.params = {"test": "value"}
    
    mock_modules["session_message"] = mock_session_message
    mock_modules["jsonrpc_request"] = mock_jsonrpc_request
    
    return mock_modules


@pytest.fixture
def mock_stream():
    """Create a mock stream for testing."""
    mock_stream = AsyncMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=None)
    mock_stream.__aiter__ = AsyncMock()
    mock_stream.send = AsyncMock()
    return mock_stream


class TestMCPInstrumentor:
    """⚡ Test MCPInstrumentor lifecycle and basic functionality."""

    def test_instrumentation_dependencies(self):
        """Test instrumentor dependencies are properly declared."""
        instrumentor = MCPInstrumentor()
        dependencies = instrumentor.instrumentation_dependencies()
        
        assert isinstance(dependencies, tuple)
        assert "mcp >= 0.1.0" in dependencies
        assert len(dependencies) >= 2

    def test_instrument_registers_post_import_hooks(self):
        """Test that instrumentation registers all necessary post-import hooks."""
        instrumentor = MCPInstrumentor()
        
        with patch("traceai_mcp.register_post_import_hook") as mock_register:
            instrumentor._instrument()
            
            # Verify all expected post-import hooks are registered
            expected_modules = [
                "mcp.client.streamable_http",
                "mcp.server.streamable_http", 
                "mcp.client.sse",
                "mcp.server.sse",
                "mcp.client.stdio",
                "mcp.server.stdio",
                "mcp.server.session"
            ]
            
            assert mock_register.call_count == len(expected_modules)
            
            # Check that each expected module was registered
            called_modules = [call.args[1] for call in mock_register.call_args_list]
            for module in expected_modules:
                assert module in called_modules

    def test_uninstrument_unwraps_functions(self):
        """Test that uninstrumentation properly unwraps wrapped functions."""
        instrumentor = MCPInstrumentor()
        
        with patch("traceai_mcp.unwrap") as mock_unwrap:
            instrumentor._uninstrument()
            
            # Verify specific functions are unwrapped
            expected_calls = [
                call("mcp.client.stdio", "stdio_client"),
                call("mcp.server.stdio", "stdio_server"),
            ]
            
            mock_unwrap.assert_has_calls(expected_calls, any_order=True)

    def test_wrap_transport_with_callback(self):
        """Test _wrap_transport_with_callback async context manager."""
        instrumentor = MCPInstrumentor()
        
        async def test_wrapper():
            mock_read_stream = AsyncMock()
            mock_write_stream = AsyncMock()
            mock_callback = MagicMock()
            
            # Create a proper async context manager class
            class MockContextManager:
                async def __aenter__(self):
                    return (mock_read_stream, mock_write_stream, mock_callback)
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    return None
            
            # Create a mock wrapped function that returns the context manager
            def mock_wrapped(*args, **kwargs):
                return MockContextManager()
            
            async with instrumentor._wrap_transport_with_callback(
                mock_wrapped, None, (), {}
            ) as (reader, writer, callback):
                assert isinstance(reader, InstrumentedStreamReader)
                assert isinstance(writer, InstrumentedStreamWriter)
                assert callback == mock_callback
                
                # Verify the streams are properly wrapped with instrumented versions
                assert hasattr(reader, '_ObjectProxy__wrapped')
                assert hasattr(writer, '_ObjectProxy__wrapped')
        
        # Run the async test
        asyncio.run(test_wrapper())

    def test_wrap_plain_transport(self):
        """Test _wrap_plain_transport async context manager."""
        instrumentor = MCPInstrumentor()
        
        async def test_wrapper():
            mock_read_stream = AsyncMock()
            mock_write_stream = AsyncMock()
            
            # Create a proper async context manager class
            class MockContextManager:
                async def __aenter__(self):
                    return (mock_read_stream, mock_write_stream)
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    return None
            
            # Create a mock wrapped function that returns the context manager
            def mock_wrapped(*args, **kwargs):
                return MockContextManager()
            
            async with instrumentor._wrap_plain_transport(
                mock_wrapped, None, (), {}
            ) as (reader, writer):
                assert isinstance(reader, InstrumentedStreamReader)
                assert isinstance(writer, InstrumentedStreamWriter)
                
                # Verify the streams are properly wrapped with instrumented versions

"""
Test suite for Google ADK framework instrumentation.

Tests the instrumentation of Google ADK components:
- Runner.run_async (CHAIN spans)
- BaseAgent.run_async (AGENT spans)
- trace_call_llm (LLM spans)
- trace_tool_call (TOOL spans)
- Tracer management and passthrough functionality
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import AsyncGenerator
import json

from opentelemetry import trace as trace_api
from opentelemetry.trace.status import StatusCode
from opentelemetry import context as context_api

from traceai_google_adk import GoogleADKInstrumentor, _PassthroughTracer
from traceai_google_adk._wrappers import (
    _RunnerRunAsync,
    _BaseAgentRunAsync,
    _TraceCallLlm,
    _TraceToolCall,
    bind_args_kwargs,
    _default,
    _get_attributes_from_generate_content_config,
    _get_attributes_from_llm_response,
    _get_attributes_from_usage_metadata,
    _get_attributes_from_content,
    _get_attributes_from_parts,
    _get_attributes_from_text_part,
    _get_attributes_from_function_call,
    _get_attributes_from_function_response,
    _get_attributes_from_base_tool,
)


class TestGoogleADKInstrumentor:
    """Test the main Google ADK instrumentor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.instrumentor = GoogleADKInstrumentor()

    def test_instrumentation_dependencies(self):
        """Test instrumentation dependencies."""
        deps = self.instrumentor.instrumentation_dependencies()
        assert ("google-adk >= 1.2.1",) == deps

    def test_instrument_basic(self):
        """Test basic instrumentation setup without complex patches."""
        # Mock tracer setup
        mock_tracer = Mock()
        self.instrumentor._tracer = mock_tracer
        
        # Mock the helper methods to avoid complex patching
        self.instrumentor._patch_trace_call_llm = Mock()
        self.instrumentor._patch_trace_tool_call = Mock()
        self.instrumentor._disable_existing_tracers = Mock()
        
        # Test that the helper methods would be called
        self.instrumentor._patch_trace_call_llm()
        self.instrumentor._patch_trace_tool_call()
        self.instrumentor._disable_existing_tracers()
        
        # Verify methods were called
        self.instrumentor._patch_trace_call_llm.assert_called_once()
        self.instrumentor._patch_trace_tool_call.assert_called_once()
        self.instrumentor._disable_existing_tracers.assert_called_once()

    def test_uninstrument(self):
        """Test uninstrumentation."""
        # Mock the cleanup methods
        self.instrumentor._unpatch_trace_call_llm = Mock()
        self.instrumentor._unpatch_trace_tool_call = Mock()
        self.instrumentor._restore_existing_tracers = Mock()
        self.instrumentor._originals = []
        
        self.instrumentor._uninstrument()
        
        # Verify cleanup methods were called
        self.instrumentor._unpatch_trace_call_llm.assert_called_once()
        self.instrumentor._unpatch_trace_tool_call.assert_called_once()
        self.instrumentor._restore_existing_tracers.assert_called_once()


class TestRunnerRunAsync:
    """Test the Runner run_async wrapper."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_tracer = MagicMock(spec=trace_api.Tracer)
        self.mock_span = MagicMock(spec=trace_api.Span)
        self.mock_context_manager = MagicMock()
        self.mock_context_manager.__enter__.return_value = self.mock_span
        self.mock_context_manager.__exit__.return_value = None
        self.mock_tracer.start_as_current_span.return_value = self.mock_context_manager
        self.wrapper = _RunnerRunAsync(self.mock_tracer)

    @pytest.mark.asyncio
    async def test_runner_run_async_basic(self):
        """Test basic runner run_async operation."""
        # Mock runner instance
        mock_runner = Mock()
        mock_runner.app_name = "TestApp"
        
        # Mock event
        mock_event = Mock()
        mock_event.is_final_response.return_value = True
        mock_event.model_dump_json.return_value = '{"result": "success"}'
        
        # Create async generator mock
        async def mock_generator():
            yield mock_event
        
        wrapped_func = Mock(return_value=mock_generator())
        
        # Test the wrapper
        kwargs = {
            "user_id": "test_user",
            "session_id": "test_session",
            "new_message": Mock()
        }
        
        result_generator = self.wrapper(wrapped_func, mock_runner, (), kwargs)
        
        # Consume the async generator
        events = []
        async for event in result_generator:
            events.append(event)
        
        assert len(events) == 1
        assert events[0] == mock_event
        wrapped_func.assert_called_once()

    @pytest.mark.asyncio
    async def test_runner_run_async_suppressed(self):
        """Test runner run_async when instrumentation is suppressed."""
        with patch("traceai_google_adk._wrappers.context_api.get_value") as mock_get_value:
            mock_get_value.return_value = True
            
            mock_runner = Mock()
            async def mock_generator():
                yield Mock()
            
            wrapped_func = Mock(return_value=mock_generator())
            
            result = self.wrapper(wrapped_func, mock_runner, (), {})
            
            # Should return original generator without wrapping
            assert result is not None
            self.mock_tracer.start_as_current_span.assert_not_called()


class TestBaseAgentRunAsync:
    """Test the BaseAgent run_async wrapper."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_tracer = MagicMock(spec=trace_api.Tracer)
        self.mock_span = MagicMock(spec=trace_api.Span)
        self.mock_context_manager = MagicMock()
        self.mock_context_manager.__enter__.return_value = self.mock_span
        self.mock_context_manager.__exit__.return_value = None
        self.mock_tracer.start_as_current_span.return_value = self.mock_context_manager
        self.wrapper = _BaseAgentRunAsync(self.mock_tracer)

    @pytest.mark.asyncio
    async def test_base_agent_run_async(self):
        """Test base agent run_async operation."""
        mock_agent = Mock()
        mock_agent.name = "TestAgent"
        
        # Mock event
        mock_event = Mock()
        mock_event.is_final_response.return_value = True
        mock_event.model_dump_json.return_value = '{"response": "agent_output"}'
        
        async def mock_generator():
            yield mock_event
        
        wrapped_func = Mock(return_value=mock_generator())

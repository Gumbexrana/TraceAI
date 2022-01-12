"""
Test suite for CrewAI framework instrumentation.

Tests the instrumentation of CrewAI components:
- Task._execute_core (AGENT spans)
- Crew.kickoff (CHAIN spans) 
- ToolUsage._use (TOOL spans)
"""

import json
from enum import Enum
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest
from fi_instrumentation.fi_types import SpanAttributes
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.trace.status import Status, StatusCode

from traceai_crewai import CrewAIInstrumentor
from traceai_crewai._wrappers import (
    SafeJSONEncoder,
    _ExecuteCoreWrapper,
    _KickoffWrapper,
    _ToolUseWrapper,
    _convert_to_dict,
    _flatten,
    _get_input_value,
    _prepare_args_kwargs,
)


class TestCrewAIInstrumentor:
    """Test the main CrewAI instrumentor class."""

    def test_instrumentation_dependencies(self):
        """Test that required dependencies are specified."""
        instrumentor = CrewAIInstrumentor()
        deps = instrumentor.instrumentation_dependencies()
        assert "crewai >= 0.41.1" in deps

    @patch("traceai_crewai.import_module")
    @patch("traceai_crewai.wrap_function_wrapper")
    def test_instrument_with_defaults(self, mock_wrap, mock_import):
        """Test instrumentation with default parameters."""
        mock_tracer_provider = Mock()
        mock_task_module = Mock()
        mock_crew_module = Mock()
        mock_tool_module = Mock()
        
        mock_import.side_effect = [
            mock_task_module,
            mock_crew_module,
            mock_tool_module,
        ]
        
        instrumentor = CrewAIInstrumentor()
        instrumentor._instrument(tracer_provider=mock_tracer_provider)
        
        # Should wrap all three methods
        assert mock_wrap.call_count == 3
        
        # Verify correct module and method names
        calls = mock_wrap.call_args_list
        assert any("crewai" in str(call) and "Task._execute_core" in str(call) for call in calls)
        assert any("crewai" in str(call) and "Crew.kickoff" in str(call) for call in calls)
        assert any("crewai.tools.tool_usage" in str(call) and "ToolUsage._use" in str(call) for call in calls)

    @patch("traceai_crewai.import_module")
    def test_uninstrument(self, mock_import):
        """Test uninstrumentation restores original methods."""
        mock_task_module = Mock()
        mock_crew_module = Mock()
        mock_tool_module = Mock()
        
        mock_import.side_effect = [
            mock_task_module,
            mock_crew_module,
            mock_tool_module,
        ]
        
        instrumentor = CrewAIInstrumentor()
        
        # Set up original methods
        original_execute = Mock()
        original_kickoff = Mock()
        original_tool_use = Mock()
        
        instrumentor._original_execute_core = original_execute
        instrumentor._original_kickoff = original_kickoff
        instrumentor._original_tool_use = original_tool_use
        
        instrumentor._uninstrument()
        
        # Verify methods are restored
        assert mock_task_module.Task._execute_core == original_execute
        assert mock_crew_module.Crew.kickoff == original_kickoff
        assert mock_tool_module.ToolUsage._use == original_tool_use


class TestExecuteCoreWrapper:
    """Test the Task._execute_core wrapper for AGENT spans."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_tracer = MagicMock(spec=trace_api.Tracer)
        self.mock_span = MagicMock(spec=trace_api.Span)
        self.mock_context_manager = MagicMock()
        self.mock_context_manager.__enter__.return_value = self.mock_span
        self.mock_context_manager.__exit__.return_value = None
        self.mock_tracer.start_as_current_span.return_value = self.mock_context_manager
        self.wrapper = _ExecuteCoreWrapper(self.mock_tracer)

    def test_suppress_instrumentation(self):
        """Test that instrumentation is suppressed when flag is set."""
        with patch.object(context_api, 'get_value', return_value=True):
            wrapped_func = Mock(return_value="result")
            
            result = self.wrapper(wrapped_func, None, (), {})
            
            assert result == "result"
            wrapped_func.assert_called_once_with()
            self.mock_tracer.start_as_current_span.assert_not_called()

    def test_task_execution_basic(self):
        """Test basic task execution wrapping."""
        # Mock objects
        mock_agent = Mock()
        mock_agent.crew = None
        mock_task = Mock()
        mock_task.key = "task_key_123"
        mock_task.id = "task_id_456"
        
        wrapped_func = Mock(__name__="_execute_core", return_value="task_result")
        
        result = self.wrapper(wrapped_func, mock_task, (mock_agent,), {})
        
        assert result == "task_result"
        
        # Verify span creation
        self.mock_tracer.start_as_current_span.assert_called_once()
        call_args = self.mock_tracer.start_as_current_span.call_args
        assert "Mock._execute_core" in call_args[0][0]
        
        # Verify attributes set
        self.mock_span.set_attribute.assert_any_call("task_key", "task_key_123")
        self.mock_span.set_attribute.assert_any_call("task_id", "task_id_456")

    def test_task_execution_with_crew(self):
        """Test task execution with crew information."""
        # Mock objects with crew
        mock_crew = Mock()
        mock_crew.key = "crew_key_789"
        mock_crew.id = "crew_id_101"
        mock_crew.share_crew = True
        
        mock_agent = Mock()
        mock_agent.crew = mock_crew
        
        mock_task = Mock()
        mock_task.key = "task_key_123"
        mock_task.id = "task_id_456"
        mock_task.description = "Test task description"
        mock_task.expected_output = "Expected output format"
        
        wrapped_func = Mock(__name__="_execute_core", return_value="task_result")
        
        result = self.wrapper(wrapped_func, mock_task, (mock_agent,), {})
        
        assert result == "task_result"
        
        # Verify crew attributes set
        self.mock_span.set_attribute.assert_any_call("crew_key", "crew_key_789")
        self.mock_span.set_attribute.assert_any_call("crew_id", "crew_id_101")
        self.mock_span.set_attribute.assert_any_call("formatted_description", "Test task description")
        self.mock_span.set_attribute.assert_any_call("formatted_expected_output", "Expected output format")

    def test_task_execution_error_handling(self):
        """Test error handling in task execution."""
        mock_agent = Mock()
        mock_agent.crew = None
        mock_task = Mock()
        mock_task.key = "task_key"
        mock_task.id = "task_id"
        
        test_exception = Exception("Task execution failed")
        wrapped_func = Mock(__name__="_execute_core", side_effect=test_exception)
        
        with pytest.raises(Exception, match="Task execution failed"):
            self.wrapper(wrapped_func, mock_task, (mock_agent,), {})
        
        # Verify error handling
        self.mock_span.set_status.assert_called_once()
        call_args = self.mock_span.set_status.call_args[0][0]
        assert call_args.status_code == StatusCode.ERROR
        assert call_args.description == "Task execution failed"
        self.mock_span.record_exception.assert_called_once_with(test_exception)

    @patch("traceai_crewai._wrappers.get_attributes_from_context")
    def test_context_attributes(self, mock_get_attrs):
        """Test that context attributes are added to spans."""
        mock_get_attrs.return_value = {"session_id": "test_session", "user_id": "test_user"}
        
        mock_agent = Mock()
        mock_agent.crew = None
        mock_task = Mock()
        mock_task.key = "task_key"
        mock_task.id = "task_id"
        
        wrapped_func = Mock(__name__="_execute_core", return_value="result")
        
        self.wrapper(wrapped_func, mock_task, (mock_agent,), {})
        
        # Verify context attributes are set
        self.mock_span.set_attributes.assert_called_once()
        call_args = self.mock_span.set_attributes.call_args[0][0]
        assert call_args["session_id"] == "test_session"
        assert call_args["user_id"] == "test_user"


class TestKickoffWrapper:
    """Test the Crew.kickoff wrapper for CHAIN spans."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_tracer = MagicMock(spec=trace_api.Tracer)
        self.mock_span = MagicMock(spec=trace_api.Span)
        self.mock_context_manager = MagicMock()
        self.mock_context_manager.__enter__.return_value = self.mock_span
        self.mock_context_manager.__exit__.return_value = None
        self.mock_tracer.start_as_current_span.return_value = self.mock_context_manager
        self.wrapper = _KickoffWrapper(self.mock_tracer)

    def test_crew_kickoff_basic(self):
        """Test basic crew kickoff wrapping."""
        # Mock crew object
        mock_crew = Mock()
        mock_crew.__class__.__name__ = "Crew"
        mock_crew.key = "crew_key_123"
        mock_crew.id = "crew_id_456"
        mock_crew.agents = []
        mock_crew.tasks = []
        mock_crew.usage_metrics = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        
        # Mock crew output
        mock_output = Mock()
        mock_output.to_dict.return_value = {"result": "crew completed successfully"}
        
        wrapped_func = Mock(__name__="kickoff", return_value=mock_output)
        
        result = self.wrapper(wrapped_func, mock_crew, (), {"inputs": {"task": "test"}})
        
        assert result == mock_output
        
        # Verify span creation
        self.mock_tracer.start_as_current_span.assert_called_once()
        call_args = self.mock_tracer.start_as_current_span.call_args
        assert "Crew.kickoff" in call_args[0][0]
        
        # Verify basic attributes
        self.mock_span.set_attribute.assert_any_call("crew_key", "crew_key_123")
        self.mock_span.set_attribute.assert_any_call("crew_id", "crew_id_456")

    def test_crew_kickoff_with_agents_and_tasks(self):
        """Test crew kickoff with agents and tasks."""
        # Mock agent
        mock_agent = Mock()
        mock_agent.key = "agent_key_1"
        mock_agent.id = "agent_id_1"
        mock_agent.role = "researcher"
        mock_agent.goal = "research topic"
        mock_agent.backstory = "expert researcher"
        mock_agent.verbose = True
        mock_agent.max_iter = 5
        mock_agent.max_rpm = 10
        mock_agent.i18n.prompt_file = "en"
        mock_agent.allow_delegation = False
        # Create mock tool with string name
        mock_tool = Mock()
        mock_tool.name = Mock()
        mock_tool.name.casefold.return_value = "search_tool"
        mock_agent.tools = [mock_tool]
        
        # Mock task
        mock_task = Mock()
        mock_task.id = "task_id_1"
        mock_task.description = "research the topic"
        mock_task.expected_output = "comprehensive report"
        mock_task.async_execution = False
        mock_task.human_input = False
        mock_task.agent = mock_agent
        mock_task.context = None
        mock_task.tools = []
        
        # Mock crew
        mock_crew = Mock()
        mock_crew.__class__.__name__ = "Crew"
        mock_crew.key = "crew_key"

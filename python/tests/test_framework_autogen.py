"""
Autogen Framework Instrumentation Tests

Tests for the Autogen framework instrumentation to verify it correctly instruments
Autogen ConversableAgent calls and generates appropriate spans and attributes.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
import json

# Import the instrumentation
from traceai_autogen import AutogenInstrumentor
from traceai_autogen.utils import _to_dict

from fi_instrumentation.otel import register
from fi_instrumentation.fi_types import (
    EvalName,
    EvalSpanKind, 
    EvalTag,
    EvalTagType,
    ProjectType,
    ModelChoices,
)
from fi_instrumentation.instrumentation.context_attributes import using_attributes


class TestAutogenFramework:
    """Test Autogen framework instrumentation."""

    @pytest.fixture(autouse=True) 
    def setup_trace_provider(self):
        """Set up trace provider for Autogen testing."""
        eval_tags = [
            EvalTag(
                eval_name=EvalName.TOXICITY,
                value=EvalSpanKind.AGENT,
                type=EvalTagType.OBSERVATION_SPAN,
                model=ModelChoices.TURING_LARGE,
                mapping={"input": "messages"},
                custom_eval_name="autogen_test_eval"
            )
        ]
        
        with patch('fi_instrumentation.otel.check_custom_eval_config_exists') as mock_check:
            mock_check.return_value = False
            self.trace_provider = register(
                project_type=ProjectType.EXPERIMENT,
                eval_tags=eval_tags,
                project_name="autogen_framework_test",
                project_version_name="v1.0",
                verbose=False
            )
        yield
        
    @pytest.fixture
    def mock_autogen_module(self):
        """Mock the autogen module and ConversableAgent."""
        mock_autogen = MagicMock()
        
        # Create mock ConversableAgent class
        class MockConversableAgent:
            def __init__(self, name="test_agent", llm_config=None):
                self.name = name
                self.llm_config = llm_config
                self._function_map = {}
                
            def generate_reply(self, messages=None, sender=None, **kwargs):
                return "This is a generated reply from the agent."
                
            def initiate_chat(self, recipient, message=None, **kwargs):
                # Mock chat result with chat_history
                result = MagicMock()
                result.chat_history = [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": "Agent response to user message"}
                ]
                return result
                
            def execute_function(self, func_call, call_id=None, verbose=False):
                if isinstance(func_call, str):
                    return f"Executed function: {func_call}"
                elif isinstance(func_call, dict):
                    func_name = func_call.get("name", "unknown")
                    return f"Executed function: {func_name} with args: {func_call.get('arguments', {})}"
                return "Function execution result"
        
        mock_autogen.ConversableAgent = MockConversableAgent
        
        with patch('importlib.import_module') as mock_import:
            mock_import.return_value = mock_autogen
            yield mock_autogen
    
    def test_autogen_import(self):
        """Test that we can import the Autogen instrumentor."""
        assert AutogenInstrumentor is not None
        
        # Test basic instantiation
        instrumentor = AutogenInstrumentor()
        assert instrumentor is not None
        assert instrumentor._original_generate is None
        assert instrumentor._original_initiate_chat is None  
        assert instrumentor._original_execute_function is None
    
    def test_autogen_basic_instrumentation(self, mock_autogen_module):
        """Test basic Autogen ConversableAgent instrumentation."""
        # Initialize instrumentor
        instrumentor = AutogenInstrumentor()
        instrumentor.instrument(tracer_provider=self.trace_provider)
        
        try:
            # Create agent and test generate_reply
            agent = mock_autogen_module.ConversableAgent(name="test_agent")
            
            with using_attributes(
                session_id="test-session-123",
                user_id="test-user-456", 
                metadata={"test_type": "instrumentation", "framework": "autogen"},
                tags=["autogen", "test", "agent"]
            ):
                response = agent.generate_reply(
                    messages=[{"role": "user", "content": "Hello, how are you?"}],
                    sender="user"
                )
            
            # Verify response
            assert response == "This is a generated reply from the agent."
            
            # Verify original methods are stored
            assert instrumentor._original_generate is not None
            assert instrumentor._original_initiate_chat is not None  
            assert instrumentor._original_execute_function is not None
            
        finally:
            # Clean up
            instrumentor.uninstrument()
    
    def test_autogen_initiate_chat(self, mock_autogen_module):
        """Test Autogen initiate_chat instrumentation."""
        instrumentor = AutogenInstrumentor()
        instrumentor.instrument(tracer_provider=self.trace_provider)
        
        try:
            # Create agents
            agent1 = mock_autogen_module.ConversableAgent(name="agent1")
            agent2 = mock_autogen_module.ConversableAgent(name="agent2")
            
            with using_attributes(
                session_id="chat-test-session",
                metadata={"chat_type": "agent_conversation"}
            ):
                result = agent1.initiate_chat(
                    recipient=agent2,
                    message="Let's discuss the weather today."
                )
            
            # Verify chat result structure
            assert hasattr(result, 'chat_history')
            assert len(result.chat_history) == 2
            assert result.chat_history[0]["content"] == "Let's discuss the weather today."
            assert result.chat_history[1]["content"] == "Agent response to user message"
            
        finally:
            instrumentor.uninstrument()
    
    def test_autogen_execute_function_string(self, mock_autogen_module):
        """Test Autogen execute_function with string input."""
        instrumentor = AutogenInstrumentor()
        instrumentor.instrument(tracer_provider=self.trace_provider)
        
        try:
            agent = mock_autogen_module.ConversableAgent(name="function_agent")
            
            with using_attributes(session_id="function-test-session"):
                result = agent.execute_function("get_weather")
            
            # Verify function execution result
            assert result == "Executed function: get_weather"
            
        finally:
            instrumentor.uninstrument()
    
    def test_autogen_execute_function_dict(self, mock_autogen_module):
        """Test Autogen execute_function with dictionary input."""
        instrumentor = AutogenInstrumentor()
        instrumentor.instrument(tracer_provider=self.trace_provider)
        

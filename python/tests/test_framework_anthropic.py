"""
Anthropic Framework Instrumentation Tests

Tests for the Anthropic framework instrumentation to verify it correctly instruments
Anthropic client calls and generates appropriate spans and attributes.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import the instrumentation
from traceai_anthropic import AnthropicInstrumentor
import anthropic

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


class TestAnthropicFramework:
    """Test Anthropic framework instrumentation."""

    @pytest.fixture(autouse=True) 
    def setup_trace_provider(self):
        """Set up trace provider for Anthropic testing."""
        eval_tags = [
            EvalTag(
                eval_name=EvalName.TOXICITY,
                value=EvalSpanKind.LLM,
                type=EvalTagType.OBSERVATION_SPAN,
                model=ModelChoices.TURING_LARGE,
                mapping={"input": "prompt"},
                custom_eval_name="anthropic_test_eval"
            )
        ]
        
        with patch('fi_instrumentation.otel.check_custom_eval_config_exists') as mock_check:
            mock_check.return_value = False
            self.trace_provider = register(
                project_type=ProjectType.EXPERIMENT,
                eval_tags=eval_tags,
                project_name="anthropic_framework_test",
                project_version_name="v1.0",
                verbose=False
            )
        yield
        
    @pytest.fixture
    def mock_anthropic_requests(self):
        """Mock Anthropic HTTP requests to avoid real API calls."""
        from anthropic.types import Message, Usage, TextBlock
        
        # Create a proper Message object to return
        mock_usage = Usage(input_tokens=10, output_tokens=12)
        mock_content = [TextBlock(type="text", text="This is a test response from Claude.")]
        mock_message = Message(
            id="msg_test123",
            type="message",
            role="assistant",
            content=mock_content,
            model="claude-3-7-sonnet-latest",
            stop_reason="end_turn",
            stop_sequence=None,
            usage=mock_usage
        )
        
        with patch('anthropic.resources.messages.Messages.create') as mock_create:
            mock_create.return_value = mock_message
            yield mock_create
    
    def test_anthropic_import(self):
        """Test that we can import the Anthropic instrumentor."""
        assert AnthropicInstrumentor is not None
        
        # Test basic instantiation
        instrumentor = AnthropicInstrumentor()
        assert instrumentor is not None
    
    def test_anthropic_basic_instrumentation(self, mock_anthropic_requests):
        """Test basic Anthropic messages instrumentation."""
        # Initialize instrumentor
        instrumentor = AnthropicInstrumentor()
        instrumentor.instrument(tracer_provider=self.trace_provider)
        
        try:
            # Create client and make request
            client = anthropic.Anthropic(api_key="test-key")
            
            with using_attributes(
                session_id="test-session-123",
                user_id="test-user-456", 
                metadata={"test_type": "instrumentation", "framework": "anthropic"},
                tags=["anthropic", "test", "claude-3-sonnet"]
            ):
                response = client.messages.create(
                    model="claude-3-7-sonnet-latest",
                    max_tokens=50,
                    messages=[
                        {"role": "user", "content": "Write a short poem about AI testing."}
                    ]
                )
            
            # Verify the create method was called
            assert mock_anthropic_requests.called

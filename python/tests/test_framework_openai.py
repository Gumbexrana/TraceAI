"""
OpenAI Framework Instrumentation Tests

Tests for the OpenAI framework instrumentation to verify it correctly instruments
OpenAI client calls and generates appropriate spans and attributes.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import the instrumentation
from traceai_openai import OpenAIInstrumentor
import openai

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


class TestOpenAIFramework:
    """Test OpenAI framework instrumentation."""

    @pytest.fixture(autouse=True) 
    def setup_trace_provider(self):
        """Set up trace provider for OpenAI testing."""
        eval_tags = [
            EvalTag(
                eval_name=EvalName.TOXICITY,
                value=EvalSpanKind.LLM,
                type=EvalTagType.OBSERVATION_SPAN,
                model=ModelChoices.TURING_LARGE,
                mapping={"input": "prompt"},
                custom_eval_name="openai_test_eval"
            )
        ]
        
        with patch('fi_instrumentation.otel.check_custom_eval_config_exists') as mock_check:
            mock_check.return_value = False
            self.trace_provider = register(
                project_type=ProjectType.EXPERIMENT,
                eval_tags=eval_tags,
                project_name="openai_framework_test",
                project_version_name="v1.0",
                verbose=False
            )
        yield
        
    @pytest.fixture
    def mock_openai_requests(self):
        """Mock OpenAI HTTP requests to avoid real API calls."""
        with patch('httpx.Client.send') as mock_request:
            # Set up mock response for chat completions
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.text = '''{"id": "chatcmpl-test123", "object": "chat.completion", "created": 1699999999, "model": "gpt-4o", "choices": [{"index": 0, "message": {"role": "assistant", "content": "This is a test response from GPT."}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25}}'''
            mock_response.json.return_value = {
                "id": "chatcmpl-test123",
                "object": "chat.completion",
                "created": 1699999999,
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a test response from GPT."
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 10,
                    "total_tokens": 25
                }
            }
            mock_request.return_value = mock_response
            yield mock_request
    
    def test_openai_import(self):
        """Test that we can import the OpenAI instrumentor."""
        assert OpenAIInstrumentor is not None
        
        # Test basic instantiation
        instrumentor = OpenAIInstrumentor()
        assert instrumentor is not None
    
    def test_openai_basic_instrumentation(self, mock_openai_requests):
        """Test basic OpenAI chat completion instrumentation."""
        # Initialize instrumentor
        instrumentor = OpenAIInstrumentor()
        instrumentor.instrument(tracer_provider=self.trace_provider)
        
        try:
            # Create client and make request
            client = openai.OpenAI(api_key="test-key")
            
            with using_attributes(
                session_id="test-session-123",
                user_id="test-user-456", 
                metadata={"test_type": "instrumentation", "framework": "openai"},
                tags=["openai", "test", "gpt-4o"]
            ):
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Write a short poem about software testing."}

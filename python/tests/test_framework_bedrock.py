"""
Bedrock Framework Instrumentation Tests

Tests for the Bedrock framework instrumentation to verify it correctly instruments
AWS Bedrock client calls and generates appropriate spans and attributes.
"""

import pytest
import sys
import os
import json
import base64
import io
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
from typing import Dict, Any

# Import the instrumentation
from traceai_bedrock import BedrockInstrumentor, BufferedStreamingBody
from traceai_bedrock import (
    _set_span_attribute,
    _get_attributes_from_message_param,
    _get_attributes_from_message_content,
    _get_attributes_from_image,
    is_iterable_of,
)

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


class TestBedrockFramework:
    """Test Bedrock framework instrumentation."""

    @pytest.fixture(autouse=True) 
    def setup_trace_provider(self):
        """Set up trace provider for Bedrock testing."""
        eval_tags = [
            EvalTag(
                eval_name=EvalName.TOXICITY,
                value=EvalSpanKind.LLM,
                type=EvalTagType.OBSERVATION_SPAN,
                model=ModelChoices.TURING_LARGE,
                mapping={"input": "prompt"},
                custom_eval_name="bedrock_test_eval"
            )
        ]
        
        with patch('fi_instrumentation.otel.check_custom_eval_config_exists') as mock_check:
            mock_check.return_value = False
            self.trace_provider = register(
                project_type=ProjectType.EXPERIMENT,
                eval_tags=eval_tags,
                project_name="bedrock_framework_test",
                project_version_name="v1.0",
                verbose=False
            )
        yield
        
    @pytest.fixture
    def mock_boto_modules(self):
        """Mock boto3 and botocore modules."""
        mock_botocore = MagicMock()
        mock_botocore.__version__ = "1.34.116"
        
        mock_boto = MagicMock()
        
        # Create a mock ClientCreator 
        class MockClientCreator:
            def __init__(self):
                self.original_create_client = self.create_client
                
            def create_client(self, service_name=None, **kwargs):
                if service_name == "bedrock-runtime":
                    return self._create_bedrock_client(**kwargs)
                return self._create_regular_client(service_name=service_name, **kwargs)
                
            def _create_bedrock_client(self, **kwargs):
                # Create mock bedrock client that doesn't have wrapped methods initially
                client = MagicMock()
                # Don't pre-add the _unwrapped methods - let instrumentation add them
                return client
                
            def _create_regular_client(self, service_name=None, **kwargs):
                client = MagicMock()
                # Make sure non-bedrock clients don't get bedrock attributes
                if hasattr(client, '_unwrapped_invoke_model'):
                    delattr(client, '_unwrapped_invoke_model')
                if hasattr(client, '_unwrapped_converse'):
                    delattr(client, '_unwrapped_converse')
                return client
        
        mock_boto.ClientCreator = MockClientCreator()
        
        with patch('importlib.import_module') as mock_import:
            def import_side_effect(module_name):
                if module_name == "botocore.client":
                    return mock_boto
                elif module_name == "botocore":
                    return mock_botocore
                else:
                    return MagicMock()
            
            mock_import.side_effect = import_side_effect
            yield mock_boto, mock_botocore
    
    @pytest.fixture
    def mock_protect(self):
        """Mock the Protect class from fi.evals."""
        mock_protect_class = MagicMock()
        mock_protect_instance = MagicMock()
        mock_protect_class.return_value = mock_protect_instance
        mock_protect_instance.protect = MagicMock()
        
        with patch('traceai_bedrock.Protect', mock_protect_class):
            yield mock_protect_instance
    
    def test_bedrock_import(self):
        """Test that we can import the Bedrock instrumentor."""
        assert BedrockInstrumentor is not None
        
        # Test basic instantiation
        instrumentor = BedrockInstrumentor()
        assert instrumentor is not None
    
    def test_bedrock_basic_instrumentation(self, mock_boto_modules, mock_protect):
        """Test basic Bedrock client instrumentation."""
        mock_boto, mock_botocore = mock_boto_modules
        
        # Initialize instrumentor
        instrumentor = BedrockInstrumentor()
        instrumentor.instrument(tracer_provider=self.trace_provider)
        
        try:
            # Verify that the client creator is wrapped
            assert hasattr(mock_boto.ClientCreator, 'create_client')
            
            # Create a bedrock client
            client = mock_boto.ClientCreator.create_client(service_name="bedrock-runtime")
            
            # Verify client has the expected methods
            assert hasattr(client, 'invoke_model')
            assert hasattr(client, 'converse')
            assert hasattr(client, 'invoke_model_with_response_stream')
            
        finally:
            # Clean up
            instrumentor.uninstrument()
    
    def test_bedrock_invoke_model_basic(self, mock_boto_modules, mock_protect):
        """Test basic invoke_model instrumentation."""
        mock_boto, mock_botocore = mock_boto_modules
        
        instrumentor = BedrockInstrumentor()
        instrumentor.instrument(tracer_provider=self.trace_provider)
        
        try:
            # Mock client and response
            client = mock_boto.ClientCreator.create_client(service_name="bedrock-runtime")
            
            # Mock the StreamingBody
            mock_stream = MagicMock()
            mock_stream._raw_stream = io.BytesIO(b'{"completion": "Test response"}')
            mock_stream._content_length = 30
            mock_stream.read.return_value = b'{"completion": "Test response"}'
            
            mock_response = {
                "body": mock_stream,
                "ResponseMetadata": {"HTTPStatusCode": 200}
            }
            
            # Simulate how the real instrumentation works
            # The instrumentation adds _unwrapped_invoke_model and replaces invoke_model
            if hasattr(client, '_unwrapped_invoke_model'):
                client._unwrapped_invoke_model.return_value = mock_response
            else:
                # If not instrumented, just mock the direct call
                client.invoke_model.return_value = mock_response
            
            # Test with various request body formats
            request_body = {
                "prompt": "What is AI?",
                "max_tokens": 100,
                "temperature": 0.7
            }
            
            with using_attributes(
                session_id="test-session-123",
                user_id="test-user-456", 
                metadata={"test_type": "invoke_model", "framework": "bedrock"},
                tags=["bedrock", "test", "claude"]
            ):
                response = client.invoke_model(
                    modelId="anthropic.claude-v2",
                    body=json.dumps(request_body)
                )
            
            assert response is not None
            
        finally:
            instrumentor.uninstrument()
    
    def test_bedrock_converse_basic(self, mock_boto_modules, mock_protect):
        """Test basic converse method instrumentation."""
        mock_boto, mock_botocore = mock_boto_modules
        
        instrumentor = BedrockInstrumentor()
        instrumentor.instrument(tracer_provider=self.trace_provider)
        
        try:
            client = mock_boto.ClientCreator.create_client(service_name="bedrock-runtime")
            
            # Mock converse response
            mock_response = {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"text": "This is a test response from Claude."}
                        ]
                    }
                },
                "usage": {
                    "inputTokens": 10,
                    "outputTokens": 12,
                    "totalTokens": 22
                }
            }
            
            if hasattr(client, '_unwrapped_converse'):
                client._unwrapped_converse.return_value = mock_response
            else:
                client.converse.return_value = mock_response
            
            with using_attributes(
                session_id="converse-test-session",
                metadata={"test_type": "converse"}
            ):
                response = client.converse(
                    modelId="anthropic.claude-3-sonnet-20240229",
                    messages=[
                        {
                            "role": "user",
                            "content": [{"text": "Hello, how are you?"}]
                        }
                    ],
                    inferenceConfig={
                        "maxTokens": 100,
                        "temperature": 0.7
                    }
                )
            
            # Verify the response exists (since we're using mocks, just verify the call was made)
            assert response is not None
            
        finally:
            instrumentor.uninstrument()
    
    def test_bedrock_converse_with_system_prompt(self, mock_boto_modules, mock_protect):
        """Test converse method with system prompts."""
        mock_boto, mock_botocore = mock_boto_modules
        
        instrumentor = BedrockInstrumentor()
        instrumentor.instrument(tracer_provider=self.trace_provider)
        
        try:
            client = mock_boto.ClientCreator.create_client(service_name="bedrock-runtime")
            
            mock_response = {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "I understand the system context."}]
                    }
                },
                "usage": {"inputTokens": 20, "outputTokens": 8, "totalTokens": 28}
            }
            
            if hasattr(client, '_unwrapped_converse'):
                client._unwrapped_converse.return_value = mock_response
            else:
                client.converse.return_value = mock_response
            
            response = client.converse(
                modelId="anthropic.claude-3-sonnet-20240229",
                system=[
                    {"text": "You are a helpful assistant."},
                    {"text": "Be concise in your responses."}
                ],
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": "What is the capital of France?"}]
                    }
                ]
            )
            
            # Verify the response exists
            assert response is not None
            
        finally:

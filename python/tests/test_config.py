"""Tests for fi_instrumentation.instrumentation.config module."""

import os
import pytest
from unittest.mock import patch, MagicMock
from opentelemetry.context import get_current, get_value
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY

from fi_instrumentation.instrumentation.config import (
    TraceConfig,
    suppress_tracing,
    REDACTED_VALUE,
    FI_HIDE_LLM_INVOCATION_PARAMETERS,
    FI_HIDE_INPUTS,
    FI_HIDE_OUTPUTS,
    FI_HIDE_INPUT_MESSAGES,
    FI_HIDE_OUTPUT_MESSAGES,
    FI_HIDE_INPUT_IMAGES,
    FI_HIDE_INPUT_TEXT,
    FI_HIDE_OUTPUT_TEXT,
    FI_HIDE_EMBEDDING_VECTORS,
    FI_BASE64_IMAGE_MAX_LENGTH,
    is_base64_url,
)


class TestTraceConfig:
    """Test TraceConfig functionality."""

    def test_trace_config_default_values(self, clean_env):
        """Test TraceConfig with default values."""
        config = TraceConfig()
        
        assert config.hide_llm_invocation_parameters is False
        assert config.hide_inputs is False
        assert config.hide_outputs is False
        assert config.hide_input_messages is False
        assert config.hide_output_messages is False
        assert config.hide_input_images is False
        assert config.hide_input_text is False
        assert config.hide_output_text is False
        assert config.hide_embedding_vectors is False
        assert config.base64_image_max_length == 32_000

    def test_trace_config_explicit_values(self, clean_env):
        """Test TraceConfig with explicitly set values."""
        config = TraceConfig(
            hide_llm_invocation_parameters=True,
            hide_inputs=True,
            hide_outputs=True,
            hide_input_messages=True,
            hide_output_messages=True,
            hide_input_images=True,
            hide_input_text=True,
            hide_output_text=True,
            hide_embedding_vectors=True,
            base64_image_max_length=1000,
        )
        
        assert config.hide_llm_invocation_parameters is True
        assert config.hide_inputs is True
        assert config.hide_outputs is True
        assert config.hide_input_messages is True
        assert config.hide_output_messages is True
        assert config.hide_input_images is True
        assert config.hide_input_text is True
        assert config.hide_output_text is True
        assert config.hide_embedding_vectors is True
        assert config.base64_image_max_length == 1000

    def test_trace_config_env_variables(self, clean_env):
        """Test TraceConfig reading from environment variables."""
        os.environ[FI_HIDE_LLM_INVOCATION_PARAMETERS] = "true"
        os.environ[FI_HIDE_INPUTS] = "true"
        os.environ[FI_HIDE_OUTPUTS] = "true"
        os.environ[FI_HIDE_INPUT_MESSAGES] = "True"
        os.environ[FI_HIDE_OUTPUT_MESSAGES] = "TRUE"
        os.environ[FI_HIDE_INPUT_IMAGES] = "false"
        os.environ[FI_HIDE_INPUT_TEXT] = "false"
        os.environ[FI_HIDE_OUTPUT_TEXT] = "false"
        os.environ[FI_HIDE_EMBEDDING_VECTORS] = "False"
        os.environ[FI_BASE64_IMAGE_MAX_LENGTH] = "5000"
        
        config = TraceConfig()
        
        assert config.hide_llm_invocation_parameters is True
        assert config.hide_inputs is True
        assert config.hide_outputs is True
        assert config.hide_input_messages is True
        assert config.hide_output_messages is True
        assert config.hide_input_images is False
        assert config.hide_input_text is False
        assert config.hide_output_text is False
        assert config.hide_embedding_vectors is False
        assert config.base64_image_max_length == 5000

    def test_trace_config_env_variables_override_explicit(self, clean_env):
        """Test that explicit values override environment variables."""
        os.environ[FI_HIDE_INPUTS] = "true"
        os.environ[FI_BASE64_IMAGE_MAX_LENGTH] = "5000"
        
        config = TraceConfig(
            hide_inputs=False,
            base64_image_max_length=10000,
        )
        
        # Explicit values should override env vars
        assert config.hide_inputs is False
        assert config.base64_image_max_length == 10000


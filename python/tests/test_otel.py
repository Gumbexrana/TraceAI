"""Tests for fi_instrumentation.otel module."""

import os
import pytest
import uuid
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock, Mock
from jsonschema import ValidationError

from fi_instrumentation.otel import (
    register,
    TracerProvider,
    SimpleSpanProcessor,
    BatchSpanProcessor,
    HTTPSpanExporter,
    GRPCSpanExporter,
    Transport,
    PROJECT_NAME,
    PROJECT_TYPE,
    PROJECT_VERSION_NAME,
    PROJECT_VERSION_ID,
    EVAL_TAGS,
    METADATA,
    check_custom_eval_config_exists,
    _exporter_transport,
    _printable_headers,
    _normalized_endpoint,
)
from fi_instrumentation.fi_types import ProjectType, EvalTag, EvalName, EvalTagType, EvalSpanKind, ModelChoices


class TestConstants:
    """Test module constants."""

    def test_constants_values(self):
        """Test that constants have expected values."""
        assert PROJECT_NAME == "project_name"
        assert PROJECT_TYPE == "project_type"
        assert PROJECT_VERSION_NAME == "project_version_name"
        assert PROJECT_VERSION_ID == "project_version_id"
        assert EVAL_TAGS == "eval_tags"
        assert METADATA == "metadata"


class TestTransport:
    """Test Transport enum."""

    def test_transport_values(self):
        """Test Transport enum values."""
        assert Transport.GRPC == "grpc"
        assert Transport.HTTP == "http"

    def test_transport_is_string(self):
        """Test that Transport values are strings."""
        assert isinstance(Transport.GRPC, str)
        assert isinstance(Transport.HTTP, str)


class TestRegisterFunction:
    """Test the register function."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock external dependencies for register function."""
        with patch('fi_instrumentation.otel.check_custom_eval_config_exists') as mock_check, \
             patch('fi_instrumentation.otel.TracerProvider') as mock_provider_class, \
             patch('fi_instrumentation.otel.SimpleSpanProcessor') as mock_simple_proc, \
             patch('fi_instrumentation.otel.BatchSpanProcessor') as mock_batch_proc:
            
            mock_check.return_value = False
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider
            
            yield {
                'check': mock_check,
                'provider_class': mock_provider_class,
                'provider': mock_provider,
                'simple_proc': mock_simple_proc,
                'batch_proc': mock_batch_proc,
            }

    def test_register_basic(self, mock_dependencies, clean_env):
        """Test basic register functionality."""
        result = register()
        
        assert result is not None
        mock_dependencies['provider_class'].assert_called_once()

    def test_register_with_project_info(self, mock_dependencies, clean_env):
        """Test register with project information."""
        result = register(
            project_name="test_project",
            project_type=ProjectType.EXPERIMENT,
            project_version_name="v1.0.0"
        )
        
        # Verify TracerProvider was created with correct resource
        call_args = mock_dependencies['provider_class'].call_args
        resource = call_args[1]['resource']
        
        assert resource.attributes[PROJECT_NAME] == "test_project"
        assert resource.attributes[PROJECT_TYPE] == ProjectType.EXPERIMENT.value
        assert resource.attributes[PROJECT_VERSION_NAME] == "v1.0.0"

    def test_register_with_eval_tags(self, mock_dependencies, clean_env):
        """Test register with eval tags."""
        from fi_instrumentation.fi_types import EvalName, ModelChoices
        
        eval_tags = [
            EvalTag(
                type=EvalTagType.OBSERVATION_SPAN,
                value=EvalSpanKind.AGENT,
                eval_name=EvalName.TOXICITY,  # Use proper enum
                model=ModelChoices.TURING_LARGE,  # Required for system evals
                custom_eval_name="custom_test",
                mapping={"input": "span_input"}  # Required mapping for TOXICITY eval
            )
        ]
        
        result = register(
            project_type=ProjectType.EXPERIMENT,
            eval_tags=eval_tags
        )
        
        call_args = mock_dependencies['provider_class'].call_args
        resource = call_args[1]['resource']
        
        # Should contain serialized eval tags
        assert EVAL_TAGS in resource.attributes

    def test_register_with_metadata(self, mock_dependencies, clean_env):
        """Test register with metadata."""
        metadata = {"environment": "test", "version": "1.0"}
        
        result = register(metadata=metadata)
        
        call_args = mock_dependencies['provider_class'].call_args
        resource = call_args[1]['resource']
        
        # Should contain serialized metadata
        assert METADATA in resource.attributes

    def test_register_observe_mode(self, mock_dependencies, clean_env):
        """Test register in OBSERVE mode."""
        result = register(project_type=ProjectType.OBSERVE)

        assert result is not None
        mock_dependencies['provider_class'].assert_called_once()
        call_args = mock_dependencies['provider_class'].call_args
        resource = call_args[1]['resource']
        assert resource.attributes[PROJECT_TYPE] == ProjectType.OBSERVE.value == "observe"

    def test_register_observe_mode_validation(self, mock_dependencies, clean_env):
        """Test register validation in OBSERVE mode."""
        # Should reject eval_tags in OBSERVE mode
        with pytest.raises(Exception):  # Could be ValidationError or other implementation-specific error
            register(
                project_type=ProjectType.OBSERVE,
                eval_tags=[EvalTag(
                    type=EvalTagType.OBSERVATION_SPAN,
                    value=EvalSpanKind.AGENT,
                    eval_name=EvalName.TOXICITY,  # Use proper enum
                    model=ModelChoices.TURING_LARGE,  # Required for system evals
                    custom_eval_name="test",
                    mapping={"input": "span_input"}  # Required mapping for TOXICITY eval
                )]
            )
        
        # Should reject project_version_name in OBSERVE mode  
        with pytest.raises(Exception):  # Could be ValidationError or other implementation-specific error
            register(
                project_type=ProjectType.OBSERVE,
                project_version_name="v1.0"
            )

    def test_register_experiment_mode(self, mock_dependencies, clean_env):
        """Test register in EXPERIMENT mode.

        register() has no EXPERIMENT-only validation (only OBSERVE rejects eval tags); this used
        to pass only because the removed session_name argument raised a TypeError.
        """
        result = register(project_type=ProjectType.EXPERIMENT)

        assert result is not None
        call_args = mock_dependencies['provider_class'].call_args
        resource = call_args[1]['resource']
        assert resource.attributes[PROJECT_TYPE] == ProjectType.EXPERIMENT.value == "experiment"

    def test_register_duplicate_eval_names(self, mock_dependencies, clean_env):
        """Test register rejects duplicate custom eval names."""
        eval_tags = [
            EvalTag(
                type=EvalTagType.OBSERVATION_SPAN,
                value=EvalSpanKind.AGENT,
                eval_name=EvalName.TOXICITY,  # Use proper enum
                model=ModelChoices.TURING_LARGE,  # Required for system evals
                custom_eval_name="duplicate",
                mapping={"input": "span_input"}  # Required mapping for TOXICITY eval
            ),
            EvalTag(
                type=EvalTagType.OBSERVATION_SPAN,
                value=EvalSpanKind.AGENT,
                eval_name=EvalName.CONTENT_MODERATION,  # Use different enum
                model=ModelChoices.TURING_LARGE,  # Required for system evals
                custom_eval_name="duplicate",
                mapping={"text": "span_input"}  # Required mapping for CONTENT_MODERATION eval
            )
        ]
        
        with pytest.raises(Exception):  # Could be ValidationError or other implementation-specific error
            register(eval_tags=eval_tags)

    def test_register_existing_custom_eval_config(self, mock_dependencies, clean_env):
        """Test register rejects existing custom eval config."""
        mock_dependencies['check'].return_value = True
        
        eval_tags = [EvalTag(
            type=EvalTagType.OBSERVATION_SPAN,
            value=EvalSpanKind.AGENT,
            eval_name=EvalName.TOXICITY,  # Use proper enum
            model=ModelChoices.TURING_LARGE,  # Required for system evals
            custom_eval_name="test",
            mapping={"input": "span_input"}  # Required mapping for TOXICITY eval
        )]
        
        with pytest.raises(Exception):  # Could be ValidationError or other implementation-specific error
            register(eval_tags=eval_tags)


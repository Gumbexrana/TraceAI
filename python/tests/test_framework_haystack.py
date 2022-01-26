"""
Test suite for Haystack framework instrumentation.

Tests the instrumentation of Haystack components:
- Pipeline.run (CHAIN spans)
- Pipeline._run_component (dynamic component wrapping)
- Component.run methods for different component types:
  - GENERATOR (LLM spans)
  - EMBEDDER (EMBEDDING spans)
  - RANKER (RERANKER spans)
  - RETRIEVER (RETRIEVER spans)
  - PROMPT_BUILDER (LLM spans)
  - UNKNOWN (CHAIN spans)
- Protect.protect (guardrail protection)
- Attribute extraction utilities
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from typing import Any, Dict, List, Sequence
from enum import auto

from opentelemetry import trace as trace_api
from opentelemetry.trace.status import StatusCode
from opentelemetry import context as context_api

from traceai_haystack import HaystackInstrumentor
from traceai_haystack._wrappers import (
    _PipelineWrapper,
    _PipelineRunComponentWrapper,
    _ComponentRunWrapper,
    ComponentType,
    _get_component_class_name,
    _get_component_span_name,
    _get_component_type,
    _get_span_kind_attributes,
    _get_input_attributes,
    _get_output_attributes,
    _get_component_output_attributes,
    _get_llm_input_message_attributes,
    _get_llm_output_message_attributes,
    _get_llm_model_attributes,
    _get_llm_token_count_attributes,
    _get_embedding_model_attributes,
    _get_embedding_attributes,
    _get_reranker_model_attributes,
    _get_reranker_request_attributes,
    _get_reranker_response_attributes,
    _get_retriever_response_attributes,
    _get_bound_arguments,
    _is_vector,
    _is_list_of_documents,
    _mask_embedding_vectors,
)


class TestHaystackInstrumentor:
    """Test the main Haystack instrumentor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.instrumentor = HaystackInstrumentor()

    def test_instrumentation_dependencies(self):
        """Test instrumentation dependencies."""
        deps = self.instrumentor.instrumentation_dependencies()
        assert deps == ("haystack-ai >= 2.9.0")

    @patch("traceai_haystack.wrap_function_wrapper")
    @patch("traceai_haystack.trace_api.get_tracer")
    def test_instrument_basic(self, mock_get_tracer, mock_wrap):
        """Test basic instrumentation setup."""
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer
        
        # Mock haystack objects
        mock_pipeline = Mock()
        mock_pipeline.run = Mock()
        mock_pipeline._run_component = Mock()
        
        with patch("traceai_haystack.haystack") as mock_haystack:
            mock_haystack.Pipeline = mock_pipeline
            
            with patch("traceai_haystack.Protect") as mock_protect:
                mock_protect.protect = Mock()
                
                self.instrumentor._instrument()
        
        # Verify tracer setup
        mock_get_tracer.assert_called_once()
        
        # Verify function wrapping
        assert mock_wrap.call_count >= 3  # Pipeline.run, Pipeline._run_component, Protect.protect
        
        # Check that originals are stored
        assert hasattr(self.instrumentor, '_original_pipeline_run')
        assert hasattr(self.instrumentor, '_original_pipeline_run_component')
        assert hasattr(self.instrumentor, '_original_component_run_methods')

    def test_uninstrument(self):
        """Test uninstrumentation."""
        # Set up mock originals
        mock_original_run = Mock()
        mock_original_run_component = Mock()
        self.instrumentor._original_pipeline_run = mock_original_run
        self.instrumentor._original_pipeline_run_component = mock_original_run_component
        self.instrumentor._original_component_run_methods = {}
        
        with patch("traceai_haystack.haystack") as mock_haystack:
            mock_pipeline = Mock()
            mock_haystack.Pipeline = mock_pipeline
            
            self.instrumentor._uninstrument()
        
        # Verify originals are restored
        assert mock_pipeline.run == mock_original_run
        assert mock_pipeline._run_component == mock_original_run_component


class TestPipelineWrapper:
    """Test the pipeline wrapper."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_tracer = MagicMock(spec=trace_api.Tracer)
        self.wrapper = _PipelineWrapper(tracer=self.mock_tracer)

    def test_pipeline_wrapper_basic(self):
        """Test basic pipeline wrapper operation."""
        mock_span = MagicMock()
        self.mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        def wrapped_func(self, data=None, **kwargs):
            return {"output": "success"}
        
        mock_instance = Mock()
        result = self.wrapper(wrapped_func, mock_instance, (mock_instance,), {"data": {"input": "test"}})
        
        assert result == {"output": "success"}
        # Check that span was started with the correct name and any attributes
        args, kwargs = self.mock_tracer.start_as_current_span.call_args
        assert args[0] == "Pipeline.run"
        mock_span.set_status.assert_called_with(StatusCode.OK)

    def test_pipeline_wrapper_exception(self):
        """Test pipeline wrapper exception handling."""
        mock_span = MagicMock()
        self.mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        def wrapped_func(self, data=None, **kwargs):
            raise ValueError("Pipeline error")
        
        mock_instance = Mock()
        
        with pytest.raises(ValueError, match="Pipeline error"):
            self.wrapper(wrapped_func, mock_instance, (mock_instance,), {})
        
        # Check that error status was set
        status_call = mock_span.set_status.call_args[0][0]
        assert status_call.status_code == trace_api.StatusCode.ERROR
        assert "Pipeline error" in str(status_call.description)

    def test_pipeline_wrapper_suppressed(self):
        """Test pipeline wrapper when instrumentation is suppressed."""
        with patch("traceai_haystack._wrappers.context_api.get_value") as mock_get_value:
            mock_get_value.return_value = True
            
            def wrapped_func(self, data=None, **kwargs):
                return {"output": "bypassed"}
            
            mock_instance = Mock()
            result = self.wrapper(wrapped_func, mock_instance, (mock_instance,), {})
            
            assert result == {"output": "bypassed"}
            # Should not start span when suppressed
            self.mock_tracer.start_as_current_span.assert_not_called()


class TestPipelineRunComponentWrapper:
    """Test the pipeline run component wrapper."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_tracer = Mock()
        self.mock_wrap_method = Mock()
        self.wrapper = _PipelineRunComponentWrapper(
            tracer=self.mock_tracer,
            wrap_component_run_method=self.mock_wrap_method
        )

    def test_pipeline_run_component_wrapper_basic(self):
        """Test basic pipeline run component wrapper operation."""
        # Mock component
        mock_component_instance = Mock()
        mock_component_instance.__class__ = Mock()
        mock_component_instance.run = Mock()
        
        def wrapped_func(self, component=None, **kwargs):
            return {"result": "success"}
        
        mock_instance = Mock()
        
        result = self.wrapper(
            wrapped_func, 
            mock_instance, 
            (mock_instance,), 
            {"component": {"instance": mock_component_instance}}
        )
        
        assert result == {"result": "success"}
        # Should wrap the component run method
        self.mock_wrap_method.assert_called_once_with(
            mock_component_instance.__class__, 
            mock_component_instance.run
        )

    def test_pipeline_run_component_wrapper_no_component(self):
        """Test pipeline run component wrapper with no component."""
        def wrapped_func(self, **kwargs):
            return {"result": "success"}
        
        mock_instance = Mock()
        result = self.wrapper(wrapped_func, mock_instance, (mock_instance,), {})
        
        assert result == {"result": "success"}
        # Should not wrap anything if no component
        self.mock_wrap_method.assert_not_called()


class TestComponentRunWrapper:
    """Test the component run wrapper."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_tracer = MagicMock(spec=trace_api.Tracer)
        self.wrapper = _ComponentRunWrapper(tracer=self.mock_tracer)

    def test_component_run_wrapper_generator(self):
        """Test component run wrapper for GENERATOR components."""
        mock_span = MagicMock()
        self.mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        # Mock generator component
        mock_component = Mock()
        mock_component.__class__.__name__ = "OpenAIGenerator"
        
        def wrapped_func(self, messages=None, **kwargs):
            return {
                "replies": ["Generated response"],
                "meta": [{"model": "gpt-4", "usage": {"prompt_tokens": 10, "completion_tokens": 20}}]
            }
        
        with patch("traceai_haystack._wrappers._get_component_type") as mock_get_type:
            mock_get_type.return_value = ComponentType.GENERATOR
            
            result = self.wrapper(wrapped_func, mock_component, (mock_component,), {"messages": []})
        
        assert result["replies"] == ["Generated response"]
        mock_span.set_status.assert_called_with(StatusCode.OK)

    def test_component_run_wrapper_embedder(self):
        """Test component run wrapper for EMBEDDER components."""
        mock_span = MagicMock()
        self.mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        # Mock embedder component
        mock_component = Mock()
        mock_component.__class__.__name__ = "SentenceTransformersTextEmbedder"
        mock_component.model = "sentence-transformers/all-MiniLM-L6-v2"
        
        def wrapped_func(self, text=None, **kwargs):
            return {"embedding": [0.1, 0.2, 0.3, 0.4]}
        
        with patch("traceai_haystack._wrappers._get_component_type") as mock_get_type:
            mock_get_type.return_value = ComponentType.EMBEDDER
            
            result = self.wrapper(wrapped_func, mock_component, (mock_component,), {"text": "test text"})
        
        assert result["embedding"] == [0.1, 0.2, 0.3, 0.4]
        mock_span.set_status.assert_called_with(StatusCode.OK)

    def test_component_run_wrapper_ranker(self):
        """Test component run wrapper for RANKER components."""

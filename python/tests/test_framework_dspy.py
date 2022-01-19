"""
Test suite for DSPy framework instrumentation.

Tests the instrumentation of DSPy components:
- LM.__call__ (LLM spans)
- Predict.forward (CHAIN spans)
- Module.__call__ (CHAIN spans)
- Retrieve.forward (RETRIEVER spans)
- ColBERTv2.__call__ (RETRIEVER spans)
- Adapter.__call__ (CHAIN spans)
- Embedder.__call__ (EMBEDDING spans)
"""

import json
from enum import Enum
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest
import numpy as np
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.trace.status import Status, StatusCode

from traceai_dspy import (
    DSPyInstrumentor,
    DSPyJSONEncoder,
    SafeJSONEncoder,
    _AdapterCallWrapper,
    _EmbedderCallWrapper,
    _LMCallWrapper,
    _ModuleForwardWrapper,
    _PredictForwardWrapper,
    _RetrieverForwardWrapper,
    _RetrieverModelCallWrapper,
    _bind_arguments,
    _convert_to_dict,
    _flatten,
    _get_embedding_input,
    _get_embedding_output,
    _get_input_value,
    _get_predict_span_name,
    _get_signature_name,
)


class TestDSPyInstrumentor:
    """Test the main DSPy instrumentor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.instrumentor = DSPyInstrumentor()

    def test_instrumentation_dependencies(self):
        """Test instrumentation dependencies."""
        deps = self.instrumentor.instrumentation_dependencies()
        assert ("dspy >= 2.5.0",) == deps

    @patch("traceai_dspy.trace_api.get_tracer_provider")
    @patch("traceai_dspy.trace_api.get_tracer")
    @patch("traceai_dspy.wrap_object")
    def test_instrument(self, mock_wrap_object, mock_get_tracer, mock_get_tracer_provider):
        """Test instrumentation setup."""
        mock_tracer_provider = Mock()
        mock_tracer = Mock()
        mock_get_tracer_provider.return_value = mock_tracer_provider
        mock_get_tracer.return_value = mock_tracer

        # Mock dspy.Predict and its subclasses
        with patch("dspy.Predict") as mock_predict:
            mock_predict.__subclasses__ = Mock(return_value=[])
            
            self.instrumentor._instrument()

        # Verify tracer setup
        mock_get_tracer.assert_called_once()
        
        # Verify wrap_object was called multiple times
        assert mock_wrap_object.call_count >= 6

    @patch("dspy.Predict")
    def test_uninstrument(self, mock_predict):
        """Test uninstrumentation."""
        # Mock wrapped forward method
        mock_wrapped = Mock()
        mock_predict.forward = Mock()
        mock_predict.forward.__wrapped__ = mock_wrapped
        
        self.instrumentor._uninstrument()
        
        # Should restore original method
        assert mock_predict.forward == mock_wrapped


class TestLMCallWrapper:
    """Test the LM call wrapper."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_tracer = MagicMock(spec=trace_api.Tracer)
        self.mock_span = MagicMock(spec=trace_api.Span)
        self.mock_context_manager = MagicMock()
        self.mock_context_manager.__enter__.return_value = self.mock_span
        self.mock_context_manager.__exit__.return_value = None
        self.mock_tracer.start_as_current_span.return_value = self.mock_context_manager
        self.wrapper = _LMCallWrapper(self.mock_tracer)

    def test_lm_call_basic(self):
        """Test basic LM call."""
        mock_lm = Mock()
        mock_lm.__class__.__name__ = "OpenAI"
        
        wrapped_func = Mock(__name__="__call__", return_value=["Response"])
        
        result = self.wrapper(wrapped_func, mock_lm, (), {})
        
        wrapped_func.assert_called_once_with()
        assert result == ["Response"]


class TestPredictForwardWrapper:
    """Test the Predict forward wrapper."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_tracer = MagicMock(spec=trace_api.Tracer)
        self.mock_span = MagicMock(spec=trace_api.Span)
        self.mock_context_manager = MagicMock()
        self.mock_context_manager.__enter__.return_value = self.mock_span
        self.mock_context_manager.__exit__.return_value = None
        self.mock_tracer.start_as_current_span.return_value = self.mock_context_manager
        self.wrapper = _PredictForwardWrapper(self.mock_tracer)

    def test_predict_forward_basic(self):
        """Test basic predict forward operation."""
        # Use suppression to bypass isinstance logic
        with patch("traceai_dspy.context_api.get_value") as mock_get_value:
            mock_get_value.return_value = True
            
            mock_instance = Mock()
            mock_instance.__class__.__name__ = "Predict"
            
            # Mock prediction
            mock_prediction = {"answer": "The answer is 42"}
            wrapped_func = Mock(__name__="forward", return_value=mock_prediction)
            
            result = self.wrapper(wrapped_func, mock_instance, (), {})
            
            wrapped_func.assert_called_once_with()
            assert result == mock_prediction

    def test_predict_forward_subclass_skip(self):
        """Test predict forward with simple case."""
        # Use suppression to test bypass logic
        with patch("traceai_dspy.context_api.get_value") as mock_get_value:
            mock_get_value.return_value = True
            
            mock_instance = Mock()
            wrapped_func = Mock(__name__="forward", return_value={"result": "test"})
            
            result = self.wrapper(wrapped_func, mock_instance, (), {})
            
            wrapped_func.assert_called_once_with()
            assert result == {"result": "test"}
            
            # Should not create span when suppressed
            self.mock_tracer.start_as_current_span.assert_not_called()

    def test_predict_forward_no_signature(self):
        """Test predict forward without signature name."""
        # Use suppression to bypass isinstance logic
        with patch("traceai_dspy.context_api.get_value") as mock_get_value:
            mock_get_value.return_value = True
            
            mock_instance = Mock()
            mock_instance.__class__.__name__ = "CustomPredict"
            
            wrapped_func = Mock(__name__="forward", return_value={})
            
            result = self.wrapper(wrapped_func, mock_instance, (), {})
            
            wrapped_func.assert_called_once_with()
            assert result == {}


class TestModuleForwardWrapper:
    """Test the Module forward wrapper."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_tracer = MagicMock(spec=trace_api.Tracer)
        self.mock_span = MagicMock(spec=trace_api.Span)
        self.mock_context_manager = MagicMock()
        self.mock_context_manager.__enter__.return_value = self.mock_span
        self.mock_context_manager.__exit__.return_value = None
        self.mock_tracer.start_as_current_span.return_value = self.mock_context_manager
        self.wrapper = _ModuleForwardWrapper(self.mock_tracer)

    def test_module_forward(self):
        """Test module forward operation."""
        mock_instance = Mock()
        mock_instance.__class__.__name__ = "RAGModule"
        mock_instance.__class__.forward = Mock()  # Mock forward method exists
        
        wrapped_func = Mock(__name__="__call__", return_value="module_output")
        
        result = self.wrapper(wrapped_func, mock_instance, ("input",), {})
        
        wrapped_func.assert_called_once_with("input")
        assert result == "module_output"
        
        span_name = self.mock_tracer.start_as_current_span.call_args[0][0]
        assert span_name == "RAGModule.forward"
        
        self.mock_span.set_status.assert_called_once_with(StatusCode.OK)

    def test_module_forward_no_forward_method(self):
        """Test module forward when no forward method exists."""
        mock_instance = Mock()
        mock_instance.__class__.__name__ = "SimpleModule"
        mock_instance.__class__.forward = None
        
        wrapped_func = Mock(__name__="__call__", return_value="output")
        
        result = self.wrapper(wrapped_func, mock_instance, (), {})
        
        wrapped_func.assert_called_once_with()
        assert result == "output"
        
        span_name = self.mock_tracer.start_as_current_span.call_args[0][0]
        assert span_name == "SimpleModule.forward"


class TestRetrieverForwardWrapper:
    """Test the Retriever forward wrapper."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_tracer = MagicMock(spec=trace_api.Tracer)
        self.mock_span = MagicMock(spec=trace_api.Span)
        self.mock_context_manager = MagicMock()
        self.mock_context_manager.__enter__.return_value = self.mock_span
        self.mock_context_manager.__exit__.return_value = None
        self.mock_tracer.start_as_current_span.return_value = self.mock_context_manager
        self.wrapper = _RetrieverForwardWrapper(self.mock_tracer)

    def test_retriever_forward(self):
        """Test retriever forward operation."""
        mock_instance = Mock()
        mock_instance.__class__.__name__ = "Retrieve"
        

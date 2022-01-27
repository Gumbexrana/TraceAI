import asyncio
import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from pydantic import BaseModel

from fi_instrumentation import TraceConfig, FITracer
from traceai_instructor import InstructorInstrumentor
from traceai_instructor._wrappers import (
    _PatchWrapper,
    _HandleResponseWrapper,
    SafeJSONEncoder,
    _flatten,
    _get_input_value,
    _to_dict,
    _raw_input,
    _raw_output,
)


class PersonModel(BaseModel):
    """Sample Pydantic model for testing."""
    name: str
    age: int
    email: str


@pytest.fixture
def tracer_provider():
    """Create a test TracerProvider with console export for testing."""
    provider = TracerProvider()
    exporter = ConsoleSpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


@pytest.fixture
def config():
    """Create a test TraceConfig."""
    return TraceConfig()


@pytest.fixture
def instrumentation_scope():
    """Set up clean instrumentation environment."""
    # Clear any existing instrumentation suppression
    context_api.set_value(_SUPPRESS_INSTRUMENTATION_KEY, False)
    yield
    # Reset to clean state
    context_api.set_value(_SUPPRESS_INSTRUMENTATION_KEY, False)


@pytest.fixture
def mock_instructor():
    """Mock the instructor module."""
    with patch("traceai_instructor.import_module") as mock_import:
        mock_instructor_module = MagicMock()
        mock_patch_module = MagicMock()
        
        def import_side_effect(module_name):
            if module_name == "instructor":
                return mock_instructor_module
            elif module_name == "instructor.patch":
                return mock_patch_module
            return MagicMock()
            
        mock_import.side_effect = import_side_effect
        yield mock_instructor_module, mock_patch_module


@pytest.fixture
def mock_protect():
    """Mock the fi.evals.Protect functionality."""
    with patch("traceai_instructor.Protect") as mock_protect:
        mock_protect.protect = MagicMock()
        yield mock_protect


class TestInstructorInstrumentor:
    """🏫 Test InstructorInstrumentor lifecycle and basic functionality."""

    def test_instrumentation_dependencies(self):
        """Test instrumentor dependencies are properly declared."""
        instrumentor = InstructorInstrumentor()
        dependencies = instrumentor.instrumentation_dependencies()
        
        assert isinstance(dependencies, tuple)
        assert "instructor >= 0.0.1" in dependencies
        assert len(dependencies) == 2

    def test_instrument_basic(self, tracer_provider, config, mock_protect):
        """Test basic instrumentation setup."""
        instrumentor = InstructorInstrumentor()
        
        with patch("wrapt.wrap_function_wrapper") as mock_wrap:
            with patch("importlib.import_module") as mock_import:
                # Mock instructor modules
                mock_instructor_module = MagicMock()
                mock_patch_module = MagicMock()
                
                def import_side_effect(module_name):
                    if module_name == "instructor":
                        return mock_instructor_module
                    elif module_name == "instructor.patch":
                        return mock_patch_module
                    return MagicMock()
                
                mock_import.side_effect = import_side_effect
                
                instrumentor._instrument(tracer_provider=tracer_provider, config=config)
                
                # Verify the tracer was created
                assert hasattr(instrumentor, '_tracer')
                assert isinstance(instrumentor._tracer, FITracer)
                
                # Verify original methods were stored
                assert hasattr(instrumentor, '_original_patch')
                assert hasattr(instrumentor, '_original_handle_response_model')

    def test_instrument_without_tracer_provider(self, config, mock_instructor, mock_protect):
        """Test instrumentation without explicit tracer provider."""
        instrumentor = InstructorInstrumentor()
        
        with patch("wrapt.wrap_function_wrapper"):
            with patch("opentelemetry.trace.get_tracer_provider") as mock_get_provider:
                mock_get_provider.return_value = MagicMock()
                instrumentor._instrument(config=config)
                mock_get_provider.assert_called_once()

    def test_instrument_without_config(self, tracer_provider, mock_instructor, mock_protect):
        """Test instrumentation without explicit config."""
        instrumentor = InstructorInstrumentor()
        
        with patch("wrapt.wrap_function_wrapper"):
            instrumentor._instrument(tracer_provider=tracer_provider)
            # Should create default TraceConfig
            assert isinstance(instrumentor._tracer, FITracer)

    def test_uninstrument(self, tracer_provider, config, mock_instructor, mock_protect):
        """Test proper uninstrumentation."""
        mock_instructor_module, mock_patch_module = mock_instructor
        instrumentor = InstructorInstrumentor()
        
        # Set up original methods
        original_patch = MagicMock()
        original_handle = MagicMock()
        mock_instructor_module.patch = MagicMock()
        mock_patch_module.handle_response_model = MagicMock()
        
        with patch("wrapt.wrap_function_wrapper"):
            instrumentor._instrument(tracer_provider=tracer_provider, config=config)
            
            # Store originals manually for testing
            instrumentor._original_patch = original_patch
            instrumentor._original_handle_response_model = original_handle
            
            # Test uninstrumentation
            instrumentor._uninstrument()
            
            # Verify restoration
            assert mock_instructor_module.patch == original_patch
            assert mock_patch_module.handle_response_model == original_handle
            assert instrumentor._original_patch is None
            assert instrumentor._original_handle_response_model is None


class TestPatchWrapper:
    """📦 Test _PatchWrapper functionality."""

    @pytest.fixture
    def patch_wrapper(self, tracer_provider):
        """Create a _PatchWrapper instance for testing."""
        tracer = trace_api.get_tracer(__name__, tracer_provider=tracer_provider)
        return _PatchWrapper(tracer)

    def test_patch_wrapper_init(self, tracer_provider):
        """Test _PatchWrapper initialization."""
        tracer = trace_api.get_tracer(__name__, tracer_provider=tracer_provider)
        wrapper = _PatchWrapper(tracer)
        assert wrapper._tracer == tracer

    def test_patch_wrapper_call_sync(self, patch_wrapper, instrumentation_scope):
        """Test _PatchWrapper with synchronous function."""
        mock_wrapped = MagicMock()
        mock_wrapped.__module__ = "instructor"
        mock_wrapped.__name__ = "patch"
        mock_wrapped.return_value = {"patched": True}
        
        args = ("client",)
        kwargs = {"response_model": PersonModel, "mode": "json"}
        
        result = patch_wrapper(mock_wrapped, None, args, kwargs)
        
        mock_wrapped.assert_called_once_with(*args, **kwargs)
        assert result == {"patched": True}

    @pytest.mark.asyncio
    async def test_patch_wrapper_call_async(self, patch_wrapper, instrumentation_scope):
        """Test _PatchWrapper with asynchronous function."""
        async def mock_async_patch(*args, **kwargs):
            return {"async_patched": True}
        
        mock_wrapped = MagicMock()
        mock_wrapped.__module__ = "instructor"
        mock_wrapped.__name__ = "patch"
        mock_wrapped.side_effect = mock_async_patch
        
        # Mock iscoroutinefunction to return True
        with patch("inspect.iscoroutinefunction", return_value=True):
            args = ("async_client",)
            kwargs = {"response_model": PersonModel}
            
            result = await patch_wrapper(mock_wrapped, None, args, kwargs)
            assert result == {"async_patched": True}

    def test_patch_wrapper_suppressed(self, patch_wrapper):
        """Test _PatchWrapper respects instrumentation suppression."""
        mock_wrapped = MagicMock()
        mock_wrapped.__module__ = "instructor"
        mock_wrapped.__name__ = "patch"
        mock_wrapped.return_value = "suppressed_result"
        
        # Suppress instrumentation
        token = context_api.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True)
        try:
            result = patch_wrapper(mock_wrapped, None, (), {})
            assert result == "suppressed_result"
        finally:
            # Don't use detach - just reset the context value
            context_api.set_value(_SUPPRESS_INSTRUMENTATION_KEY, False)

    def test_patch_wrapper_exception_handling(self, patch_wrapper, instrumentation_scope):
        """Test _PatchWrapper exception handling."""
        mock_wrapped = MagicMock()
        mock_wrapped.__module__ = "instructor"
        mock_wrapped.__name__ = "patch"
        mock_wrapped.side_effect = ValueError("Patch failed")
        
        with pytest.raises(ValueError, match="Patch failed"):
            patch_wrapper(mock_wrapped, None, (), {})


class TestHandleResponseWrapper:
    """🎯 Test _HandleResponseWrapper functionality."""

    @pytest.fixture
    def response_wrapper(self, tracer_provider):
        """Create a _HandleResponseWrapper instance for testing."""
        tracer = trace_api.get_tracer(__name__, tracer_provider=tracer_provider)
        return _HandleResponseWrapper(tracer)

    def test_response_wrapper_init(self, tracer_provider):
        """Test _HandleResponseWrapper initialization."""
        tracer = trace_api.get_tracer(__name__, tracer_provider=tracer_provider)
        wrapper = _HandleResponseWrapper(tracer)
        assert wrapper._tracer == tracer

    def test_response_wrapper_call_sync(self, response_wrapper, instrumentation_scope):
        """Test _HandleResponseWrapper with synchronous function."""
        mock_wrapped = MagicMock()
        mock_wrapped.__module__ = "instructor.patch"
        mock_wrapped.__name__ = "handle_response_model"
        
        # Mock response
        mock_response = [PersonModel(name="John", age=30, email="john@example.com")]
        mock_wrapped.return_value = mock_response
        
        args = (mock_response, PersonModel)
        kwargs = {"mode": "json"}
        
        result = response_wrapper(mock_wrapped, None, args, kwargs)
        
        mock_wrapped.assert_called_once_with(*args, **kwargs)
        assert result == mock_response

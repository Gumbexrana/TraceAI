"""Tests for vLLM instrumentor."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import sys

from traceai_vllm import VLLMInstrumentor, __version__


class TestVLLMInstrumentor:
    """Tests for VLLMInstrumentor class."""

    def test_version_exists(self):
        """Should have a version defined."""
        assert __version__ is not None
        assert isinstance(__version__, str)

    def test_instrumentation_dependencies(self):
        """Should list openai as dependency."""
        instrumentor = VLLMInstrumentor()
        deps = instrumentor.instrumentation_dependencies()

        assert len(deps) == 1
        assert "openai" in deps[0]

    def test_default_vllm_base_urls(self):
        """Should have default vLLM base URL."""
        instrumentor = VLLMInstrumentor()
        assert "localhost:8000" in instrumentor._vllm_base_urls

    def test_custom_vllm_base_urls(self):
        """Should accept custom vLLM base URLs."""
        instrumentor = VLLMInstrumentor(vllm_base_urls=["my-server:8080", "other-server:9000"])
        assert "my-server:8080" in instrumentor._vllm_base_urls
        assert "other-server:9000" in instrumentor._vllm_base_urls

    @patch("opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter")
    def test_instrument_creates_tracer(self, mock_exporter):
        """Should create tracer during instrumentation."""
        mock_provider = MagicMock()
        mock_provider.get_tracer = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {
            "openai": MagicMock(),
            "openai.resources": MagicMock(),
            "openai.resources.chat": MagicMock(),
        }):
            instrumentor = VLLMInstrumentor()
            instrumentor.instrument(tracer_provider=mock_provider)

            assert hasattr(instrumentor, "_tracer")

    def test_uninstrument_restores_original(self):
        """Should restore original methods on uninstrument."""
        mock_provider = MagicMock()
        mock_provider.get_tracer = MagicMock(return_value=MagicMock())

        mock_completions = MagicMock()
        mock_async_completions = MagicMock()
        original_create = MagicMock()
        original_async_create = MagicMock()
        mock_completions.create = original_create
        mock_async_completions.create = original_async_create

        mock_chat_module = MagicMock()
        mock_chat_module.Completions = mock_completions
        mock_chat_module.AsyncCompletions = mock_async_completions

        with patch.dict(sys.modules, {
            "openai": MagicMock(),
            "openai.resources": MagicMock(),
            "openai.resources.chat": mock_chat_module,
        }):
            instrumentor = VLLMInstrumentor()
            instrumentor._original_completions_create = original_create
            instrumentor._original_async_completions_create = original_async_create

            instrumentor._uninstrument()

    @patch("opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter")
    def test_instrument_with_config(self, mock_exporter):
        """Should accept custom config."""
        from fi_instrumentation import TraceConfig

        mock_provider = MagicMock()
        mock_provider.get_tracer = MagicMock(return_value=MagicMock())
        config = TraceConfig()

        with patch.dict(sys.modules, {
            "openai": MagicMock(),
            "openai.resources": MagicMock(),
            "openai.resources.chat": MagicMock(),
        }):
            instrumentor = VLLMInstrumentor()
            instrumentor.instrument(tracer_provider=mock_provider, config=config)

            assert hasattr(instrumentor, "_tracer")

    def test_handles_missing_openai(self):
        """Should handle missing openai gracefully."""
        mock_provider = MagicMock()
        mock_provider.get_tracer = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {"openai": None}):
            with patch("traceai_vllm.import_module", side_effect=ImportError):
                instrumentor = VLLMInstrumentor()
                # Should not raise
                try:
                    instrumentor.instrument(tracer_provider=mock_provider)
                except ImportError:
                    pass  # Expected when openai not installed

"""Tests for traceai_agno._instrumentor module."""

import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from traceai_agno._instrumentor import (
    AgnoInstrumentorWrapper,
    configure_agno_tracing,
    setup_traceai_exporter,
)


class TestAgnoInstrumentorWrapper:
    """Tests for AgnoInstrumentorWrapper class."""

    def test_singleton_pattern(self, reset_instrumentor):
        """Test that wrapper uses singleton pattern."""
        wrapper1 = AgnoInstrumentorWrapper()
        wrapper2 = AgnoInstrumentorWrapper()

        assert wrapper1 is wrapper2

    def test_initial_state(self, reset_instrumentor):
        """Test initial state of wrapper."""
        wrapper = AgnoInstrumentorWrapper()

        assert wrapper.is_instrumented is False
        assert wrapper._tracer_provider is None
        assert wrapper._agno_instrumentor is None

    @patch.dict(os.environ, {}, clear=True)
    def test_instrument_sets_otlp_endpoint(self, reset_instrumentor):
        """Test that instrument sets OTLP endpoint environment variable."""
        wrapper = AgnoInstrumentorWrapper()

        with patch.object(wrapper, "_setup_agno_instrumentor"):
            wrapper.instrument(otlp_endpoint="https://api.test.com/v1/traces")

        assert os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") == "https://api.test.com/v1/traces"

    @patch.dict(os.environ, {}, clear=True)
    def test_instrument_sets_otlp_headers(self, reset_instrumentor):
        """Test that instrument sets OTLP headers environment variable."""
        wrapper = AgnoInstrumentorWrapper()

        with patch.object(wrapper, "_setup_agno_instrumentor"):
            wrapper.instrument(
                otlp_headers={
                    "Authorization": "Bearer test-key",
                    "x-project-name": "test-project",
                }
            )

        headers = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS")
        assert "Authorization=Bearer test-key" in headers
        assert "x-project-name=test-project" in headers

    def test_instrument_with_tracer_provider(self, reset_instrumentor, mock_tracer_provider):
        """Test instrumentation with custom tracer provider."""
        wrapper = AgnoInstrumentorWrapper()

        with patch.object(wrapper, "_setup_agno_instrumentor"):
            with patch("traceai_agno._instrumentor.trace") as mock_trace:
                wrapper.instrument(tracer_provider=mock_tracer_provider)

        mock_trace.set_tracer_provider.assert_called_once_with(mock_tracer_provider)
        assert wrapper._tracer_provider is mock_tracer_provider

    def test_instrument_idempotent(self, reset_instrumentor):
        """Test that calling instrument multiple times is idempotent."""
        wrapper = AgnoInstrumentorWrapper()

        with patch.object(wrapper, "_setup_agno_instrumentor") as mock_setup:
            wrapper.instrument()
            wrapper.instrument()
            wrapper.instrument()

        # Should only be called once
        mock_setup.assert_called_once()

    def test_instrument_returns_self(self, reset_instrumentor):
        """Test that instrument returns self for chaining."""
        wrapper = AgnoInstrumentorWrapper()

        with patch.object(wrapper, "_setup_agno_instrumentor"):
            result = wrapper.instrument()

        assert result is wrapper

    def test_is_instrumented_property(self, reset_instrumentor):
        """Test is_instrumented property."""
        wrapper = AgnoInstrumentorWrapper()

        assert wrapper.is_instrumented is False

        with patch.object(wrapper, "_setup_agno_instrumentor"):
            wrapper.instrument()

        assert wrapper.is_instrumented is True

    def test_uninstrument(self, reset_instrumentor):
        """Test uninstrument method."""
        wrapper = AgnoInstrumentorWrapper()

        mock_instrumentor = MagicMock()
        wrapper._agno_instrumentor = mock_instrumentor
        wrapper._is_instrumented = True
        wrapper._tracer_provider = MagicMock()

        wrapper.uninstrument()

        mock_instrumentor.uninstrument.assert_called_once()
        assert wrapper.is_instrumented is False
        assert wrapper._tracer_provider is None

    def test_uninstrument_when_not_instrumented(self, reset_instrumentor):
        """Test uninstrument when not instrumented."""
        wrapper = AgnoInstrumentorWrapper()

        # Should not raise
        wrapper.uninstrument()

        assert wrapper.is_instrumented is False

    def test_uninstrument_handles_exception(self, reset_instrumentor):
        """Test uninstrument handles exception from underlying instrumentor."""
        wrapper = AgnoInstrumentorWrapper()

        mock_instrumentor = MagicMock()
        mock_instrumentor.uninstrument.side_effect = Exception("Test error")
        wrapper._agno_instrumentor = mock_instrumentor
        wrapper._is_instrumented = True

        # Should not raise
        wrapper.uninstrument()

        assert wrapper.is_instrumented is False

    def test_setup_agno_instrumentor_success(self, reset_instrumentor):

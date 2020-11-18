"""Tests for Pydantic AI instrumentor."""

import pytest
from unittest.mock import MagicMock, patch


class TestPydanticAIInstrumentor:
    """Test PydanticAIInstrumentor class."""

    def test_singleton_pattern(self):
        """Test that instrumentor follows singleton pattern."""
        from traceai_pydantic_ai._instrumentor import PydanticAIInstrumentor

        # Reset singleton for test
        PydanticAIInstrumentor._instance = None
        PydanticAIInstrumentor._is_instrumented = False

        inst1 = PydanticAIInstrumentor()
        inst2 = PydanticAIInstrumentor()

        assert inst1 is inst2

    def test_instrumentation_dependencies(self):
        """Test instrumentation dependencies list."""
        from traceai_pydantic_ai._instrumentor import PydanticAIInstrumentor

        # Reset singleton for test
        PydanticAIInstrumentor._instance = None
        PydanticAIInstrumentor._is_instrumented = False

        inst = PydanticAIInstrumentor()
        deps = inst.instrumentation_dependencies()

        assert len(deps) > 0
        assert any("pydantic-ai" in dep for dep in deps)

    def test_is_instrumented_initially_false(self):
        """Test that is_instrumented is initially False."""
        from traceai_pydantic_ai._instrumentor import PydanticAIInstrumentor

        # Reset singleton for test
        PydanticAIInstrumentor._instance = None
        PydanticAIInstrumentor._is_instrumented = False

        inst = PydanticAIInstrumentor()
        assert inst.is_instrumented is False

    @patch("traceai_pydantic_ai._instrumentor.trace_api")
    def test_instrument_creates_tracer(self, mock_trace_api):
        """Test that instrument() creates a tracer."""
        from traceai_pydantic_ai._instrumentor import PydanticAIInstrumentor

        # Reset singleton for test
        PydanticAIInstrumentor._instance = None
        PydanticAIInstrumentor._is_instrumented = False

        mock_provider = MagicMock()
        mock_tracer = MagicMock()
        mock_trace_api.get_tracer.return_value = mock_tracer

        inst = PydanticAIInstrumentor()

        with patch.object(inst, "_patch_pydantic_ai") as mock_patch:
            with patch.object(inst, "instrumentation_dependencies", return_value=[]):
                mock_patch.side_effect = ImportError("pydantic_ai not installed")
                inst.instrument(tracer_provider=mock_provider)

        mock_trace_api.get_tracer.assert_called_once()

    @patch("traceai_pydantic_ai._instrumentor.trace_api")
    def test_instrument_skips_if_already_instrumented(self, mock_trace_api):
        """Test that instrument() is idempotent."""
        from traceai_pydantic_ai._instrumentor import PydanticAIInstrumentor

        # Reset singleton for test
        PydanticAIInstrumentor._instance = None
        PydanticAIInstrumentor._is_instrumented = False

        inst = PydanticAIInstrumentor()

        with patch.object(inst, "_patch_pydantic_ai"):
            with patch.object(inst, "instrumentation_dependencies", return_value=[]):
                inst.instrument()
                # Second call should be skipped
                inst.instrument()

        # get_tracer should only be called once
        assert mock_trace_api.get_tracer.call_count == 1

    def test_uninstrument_when_not_instrumented(self):
        """Test that uninstrument() is safe when not instrumented."""
        from traceai_pydantic_ai._instrumentor import PydanticAIInstrumentor

        # Reset singleton for test
        PydanticAIInstrumentor._instance = None
        PydanticAIInstrumentor._is_instrumented = False

        inst = PydanticAIInstrumentor()
        # Should not raise
        inst.uninstrument()

    @patch("traceai_pydantic_ai._instrumentor.trace_api")
    def test_uninstrument_clears_state(self, mock_trace_api):
        """Test that uninstrument() clears internal state."""
        from traceai_pydantic_ai._instrumentor import PydanticAIInstrumentor

        # Reset singleton for test
        PydanticAIInstrumentor._instance = None
        PydanticAIInstrumentor._is_instrumented = False

        inst = PydanticAIInstrumentor()

        with patch.object(inst, "_patch_pydantic_ai"):

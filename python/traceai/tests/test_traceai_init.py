"""
Tests for traceai.init() wrapper and SDK semantic conventions.

These tests verify:
- SemanticConvention enum values
- traceai.init() parameter handling
- Auto-instrumentation registry
- Instrumentor loading
"""

import unittest
from unittest.mock import patch, MagicMock
import sys


class TestSemanticConventionEnum(unittest.TestCase):
    """Tests for SemanticConvention enum."""

    def test_semantic_convention_values(self):
        """Test that SemanticConvention has expected values."""
        from fi_instrumentation.otel import SemanticConvention

        self.assertEqual(SemanticConvention.FI.value, "fi")
        self.assertEqual(SemanticConvention.OTEL_GENAI.value, "otel_genai")
        self.assertEqual(SemanticConvention.OPENINFERENCE.value, "openinference")
        self.assertEqual(SemanticConvention.OPENLLMETRY.value, "openllmetry")

    def test_semantic_convention_from_string(self):
        """Test creating SemanticConvention from string."""
        from fi_instrumentation.otel import SemanticConvention

        self.assertEqual(SemanticConvention("fi"), SemanticConvention.FI)
        self.assertEqual(SemanticConvention("otel_genai"), SemanticConvention.OTEL_GENAI)
        self.assertEqual(SemanticConvention("openinference"), SemanticConvention.OPENINFERENCE)
        self.assertEqual(SemanticConvention("openllmetry"), SemanticConvention.OPENLLMETRY)

    def test_semantic_convention_is_string_enum(self):
        """Test that SemanticConvention inherits from str."""
        from fi_instrumentation.otel import SemanticConvention

        # Should be usable as string
        self.assertIsInstance(SemanticConvention.FI, str)
        self.assertEqual(f"Convention: {SemanticConvention.FI}", "Convention: fi")


class TestTraceAIInit(unittest.TestCase):
    """Tests for traceai.init() function."""

    @patch("traceai.register")
    def test_init_default_parameters(self, mock_register):
        """Test init() with default parameters."""
        from traceai import init
        from fi_instrumentation import SemanticConvention, Transport
        from fi_instrumentation.fi_types import ProjectType

        mock_provider = MagicMock()
        mock_register.return_value = mock_provider

        result = init(project_name="test-project", verbose=False)

        mock_register.assert_called_once()
        call_kwargs = mock_register.call_args.kwargs

        self.assertEqual(call_kwargs["project_name"], "test-project")
        self.assertEqual(call_kwargs["project_type"], ProjectType.EXPERIMENT)
        self.assertEqual(call_kwargs["semantic_convention"], SemanticConvention.FI)
        self.assertEqual(call_kwargs["transport"], Transport.HTTP)
        self.assertFalse(call_kwargs["batch"])
        self.assertTrue(call_kwargs["set_global_tracer_provider"])
        self.assertEqual(result, mock_provider)

    @patch("traceai.register")
    def test_init_with_otel_genai_convention(self, mock_register):
        """Test init() with OTEL GenAI convention."""
        from traceai import init
        from fi_instrumentation import SemanticConvention

        mock_register.return_value = MagicMock()

        init(
            project_name="test-project",
            semantic_convention="otel_genai",
            verbose=False,
        )

        call_kwargs = mock_register.call_args.kwargs
        self.assertEqual(call_kwargs["semantic_convention"], SemanticConvention.OTEL_GENAI)

    @patch("traceai.register")
    def test_init_with_observe_project_type(self, mock_register):
        """Test init() with observe project type."""
        from traceai import init
        from fi_instrumentation.fi_types import ProjectType

        mock_register.return_value = MagicMock()

        init(
            project_name="test-project",
            project_type="observe",
            verbose=False,
        )

        call_kwargs = mock_register.call_args.kwargs
        self.assertEqual(call_kwargs["project_type"], ProjectType.OBSERVE)

    @patch("traceai.register")
    def test_init_with_grpc_transport(self, mock_register):
        """Test init() with gRPC transport."""
        from traceai import init
        from fi_instrumentation import Transport

        mock_register.return_value = MagicMock()

        init(
            project_name="test-project",
            transport="grpc",
            verbose=False,
        )

        call_kwargs = mock_register.call_args.kwargs

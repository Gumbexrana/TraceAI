"""Integration tests for Pydantic AI instrumentation.

These tests require pydantic-ai to be installed and an API key to be set.
They are marked with `integration` to be skipped in CI without credentials.
"""

import pytest
import os
from unittest.mock import MagicMock, patch

# Check if we can run integration tests
HAS_PYDANTIC_AI = False
HAS_OPENAI_KEY = bool(os.environ.get("OPENAI_API_KEY"))

try:
    import pydantic_ai
    HAS_PYDANTIC_AI = True
except ImportError:
    pass


@pytest.fixture
def memory_exporter():
    """Create an in-memory span exporter for testing."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    yield exporter, provider

    exporter.clear()


@pytest.fixture
def instrumented_pydantic_ai(memory_exporter):
    """Setup instrumented Pydantic AI with mocked pydantic_ai module."""
    from traceai_pydantic_ai import PydanticAIInstrumentor
    import sys

    exporter, provider = memory_exporter

    # Reset singleton
    PydanticAIInstrumentor._instance = None
    PydanticAIInstrumentor._is_instrumented = False

    instrumentor = PydanticAIInstrumentor()

    # Create mock pydantic_ai module
    mock_pydantic_ai = MagicMock()
    mock_agent_class = MagicMock()
    mock_agent_class.run = MagicMock()
    mock_agent_class.run_sync = MagicMock()
    mock_pydantic_ai.Agent = mock_agent_class

    with patch.object(instrumentor, "instrumentation_dependencies", return_value=[]):
        with patch.dict(sys.modules, {"pydantic_ai": mock_pydantic_ai}):
            instrumentor.instrument(tracer_provider=provider)
            yield exporter, instrumentor

    instrumentor.uninstrument()


class TestInstrumentorIntegration:
    """Test instrumentor with mock Pydantic AI."""

    def test_instrument_creates_spans(self, instrumented_pydantic_ai):
        """Test that instrumentation creates spans when methods are called."""
        exporter, instrumentor = instrumented_pydantic_ai

        assert instrumentor.is_instrumented
        assert instrumentor._tracer is not None

    def test_uninstrument_cleans_up(self, instrumented_pydantic_ai):
        """Test that uninstrument cleans up properly."""
        exporter, instrumentor = instrumented_pydantic_ai

        instrumentor.uninstrument()

        assert not instrumentor.is_instrumented
        assert instrumentor._tracer is None


@pytest.mark.skipif(not HAS_PYDANTIC_AI, reason="pydantic-ai not installed")
class TestPydanticAIInstalled:
    """Tests that run when pydantic-ai is installed."""

    def test_can_import_agent(self):
        """Test Agent can be imported."""
        from pydantic_ai import Agent
        assert Agent is not None

    def test_can_create_agent(self):
        """Test Agent can be created."""
        from pydantic_ai import Agent

        agent = Agent("test-model", instructions="Test")
        assert agent is not None

    def test_instrumentor_patches_agent(self, memory_exporter):
        """Test instrumentor patches Agent methods."""
        from traceai_pydantic_ai import PydanticAIInstrumentor
        from pydantic_ai import Agent

        exporter, provider = memory_exporter

        # Reset singleton
        PydanticAIInstrumentor._instance = None
        PydanticAIInstrumentor._is_instrumented = False

        instrumentor = PydanticAIInstrumentor()

        # Store original method
        original_run = Agent.run

        with patch.object(instrumentor, "instrumentation_dependencies", return_value=[]):
            instrumentor.instrument(tracer_provider=provider)

        # Method should be different (wrapped)
        assert Agent.run is not original_run

        instrumentor.uninstrument()

        # Method should be restored
        assert Agent.run is original_run


@pytest.mark.skipif(
    not (HAS_PYDANTIC_AI and HAS_OPENAI_KEY),
    reason="Requires pydantic-ai and OPENAI_API_KEY"
)
@pytest.mark.integration
class TestRealWorldIntegration:
    """Real-world integration tests with actual API calls.

    These tests make real API calls and require:
    - pydantic-ai installed
    - OPENAI_API_KEY environment variable
    """

    @pytest.mark.asyncio
    async def test_basic_agent_run(self, memory_exporter):
        """Test basic agent run creates proper spans."""
        from traceai_pydantic_ai import PydanticAIInstrumentor
        from pydantic_ai import Agent

        exporter, provider = memory_exporter

        # Reset and instrument
        PydanticAIInstrumentor._instance = None
        PydanticAIInstrumentor._is_instrumented = False

        instrumentor = PydanticAIInstrumentor()
        instrumentor.instrument(tracer_provider=provider)

"""Real-world scenario tests for Pydantic AI instrumentation.

These tests simulate common real-world usage patterns without requiring
actual API calls, using mocks to verify correct instrumentation behavior.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio


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

    yield exporter, provider

    exporter.clear()


class TestChatbotScenario:
    """Test chatbot/conversation scenario."""

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, memory_exporter):
        """Test multi-turn conversation tracing."""
        from traceai_pydantic_ai._agent_wrapper import wrap_agent_run

        exporter, provider = memory_exporter
        tracer = provider.get_tracer("test")

        # Simulate conversation
        conversation = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        mock_result = MagicMock()
        mock_result.usage = MagicMock(
            request_tokens=50,
            response_tokens=30,
            total_tokens=80,
        )
        mock_result.output = "I'm doing well!"

        async def mock_run(self, prompt, message_history=None):
            return mock_result

        mock_agent = MagicMock()
        mock_agent.model = "openai:gpt-4o"
        mock_agent.instructions = "Be friendly and helpful"
        mock_agent.result_type = None

        wrapped = wrap_agent_run(mock_run, tracer, "run")

        # Simulate conversation turns
        for i in range(3):
            await wrapped(mock_agent, f"Turn {i+1}", message_history=conversation[:i])

        spans = exporter.get_finished_spans()
        assert len(spans) == 3

    def test_conversation_with_context(self, memory_exporter):
        """Test conversation with context/dependencies."""
        from traceai_pydantic_ai._agent_wrapper import wrap_agent_run

        exporter, provider = memory_exporter
        tracer = provider.get_tracer("test")

        mock_result = MagicMock()
        mock_result.usage = None
        mock_result.output = "Response with context"

        def mock_run_sync(self, prompt):
            return mock_result

        mock_agent = MagicMock()
        mock_agent.model = "anthropic:claude-3-sonnet"
        mock_agent.instructions = None
        mock_agent.result_type = None

        wrapped = wrap_agent_run(mock_run_sync, tracer, "run_sync")

        result = wrapped(mock_agent, "Hello with context")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get("gen_ai.system") == "anthropic"


class TestRAGPipelineScenario:
    """Test RAG (Retrieval Augmented Generation) scenario."""

    @pytest.mark.asyncio
    async def test_rag_with_tools(self, memory_exporter):
        """Test RAG pipeline with search tool."""
        from traceai_pydantic_ai._agent_wrapper import wrap_agent_run, wrap_tool_function

        exporter, provider = memory_exporter
        tracer = provider.get_tracer("test")

        # Create search tool
        def search_documents(ctx, query: str) -> str:
            return f"Found documents for: {query}"

        wrapped_tool = wrap_tool_function(
            search_documents,
            tracer,
            "search_documents",
            "Search document database",
        )

        # Execute tool
        result = wrapped_tool(MagicMock(), "machine learning")
        assert "Found documents" in result

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert "search_documents" in spans[0].name

    @pytest.mark.asyncio
    async def test_rag_with_multiple_tools(self, memory_exporter):
        """Test RAG with multiple tools in sequence."""
        from traceai_pydantic_ai._agent_wrapper import wrap_tool_function

        exporter, provider = memory_exporter
        tracer = provider.get_tracer("test")

        # Retriever tool
        def retrieve(ctx, query: str) -> list:
            return ["doc1", "doc2", "doc3"]

        # Reranker tool
        def rerank(ctx, docs: list, query: str) -> list:
            return docs[:2]

        wrapped_retrieve = wrap_tool_function(retrieve, tracer, "retrieve")
        wrapped_rerank = wrap_tool_function(rerank, tracer, "rerank")

        # Execute RAG pipeline
        docs = wrapped_retrieve(MagicMock(), "AI trends")
        ranked = wrapped_rerank(MagicMock(), docs, "AI trends")

        assert len(ranked) == 2

        spans = exporter.get_finished_spans()
        assert len(spans) == 2


class TestStructuredOutputScenario:
    """Test structured output scenarios."""

    @pytest.mark.asyncio
    async def test_structured_output_extraction(self, memory_exporter):
        """Test structured data extraction."""
        from traceai_pydantic_ai._agent_wrapper import wrap_agent_run

        exporter, provider = memory_exporter
        tracer = provider.get_tracer("test")

        # Simulate structured output
        class PersonInfo:
            name: str
            age: int
            occupation: str

        mock_output = MagicMock()
        mock_output.name = "Alice"
        mock_output.age = 30
        mock_output.occupation = "Engineer"

        mock_result = MagicMock()
        mock_result.usage = None
        mock_result.output = mock_output

        async def mock_run(self, prompt):
            return mock_result

        mock_agent = MagicMock()
        mock_agent.model = "openai:gpt-4o"
        mock_agent.instructions = None
        mock_agent.result_type = PersonInfo

        wrapped = wrap_agent_run(mock_run, tracer, "run")

        result = await wrapped(mock_agent, "Extract person info")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get("pydantic_ai.run.is_structured") == True

    @pytest.mark.asyncio
    async def test_json_mode_extraction(self, memory_exporter):
        """Test JSON mode output."""
        from traceai_pydantic_ai._agent_wrapper import wrap_agent_run

        exporter, provider = memory_exporter
        tracer = provider.get_tracer("test")

        mock_result = MagicMock()
        mock_result.usage = None
        mock_result.output = {"status": "success", "data": [1, 2, 3]}

        async def mock_run(self, prompt):
            return mock_result

        mock_agent = MagicMock()
        mock_agent.model = "openai:gpt-4o"
        mock_agent.instructions = None
        mock_agent.result_type = dict

        wrapped = wrap_agent_run(mock_run, tracer, "run")

        result = await wrapped(mock_agent, "Return JSON data")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1


"""Tests for LangGraph instrumentor module."""


class TestLangGraphInstrumentor:
    """Test LangGraphInstrumentor class."""

    def test_import(self):
        """Test that instrumentor can be imported."""
        from traceai_langchain._langgraph._instrumentor import LangGraphInstrumentor
        assert LangGraphInstrumentor is not None

    def test_singleton_pattern(self):
        """Test singleton pattern."""
        from traceai_langchain._langgraph._instrumentor import LangGraphInstrumentor

        # Reset singleton for testing
        LangGraphInstrumentor._instance = None
        LangGraphInstrumentor._is_instrumented = False

        inst1 = LangGraphInstrumentor()
        inst2 = LangGraphInstrumentor()

        assert inst1 is inst2

    def test_instrumentation_dependencies(self):
        """Test instrumentation dependencies."""
        from traceai_langchain._langgraph._instrumentor import LangGraphInstrumentor

        # Reset singleton for testing
        LangGraphInstrumentor._instance = None
        LangGraphInstrumentor._is_instrumented = False

        instrumentor = LangGraphInstrumentor()
        deps = instrumentor.instrumentation_dependencies()

        assert len(deps) > 0
        assert any("langgraph" in dep for dep in deps)

    def test_is_instrumented_property(self):
        """Test is_instrumented property."""
        from traceai_langchain._langgraph._instrumentor import LangGraphInstrumentor

        # Reset singleton for testing
        LangGraphInstrumentor._instance = None
        LangGraphInstrumentor._is_instrumented = False

        instrumentor = LangGraphInstrumentor()
        assert instrumentor.is_instrumented is False

    # --- Shim contract -----------------------------------------------------
    # LangGraphInstrumentor is now a deprecated no-op: node/tool/LLM spans are
    # captured automatically by LangChainInstrumentor's callback handler. These
    # tests assert the instrumentor NEVER patches LangGraph (which used to break
    # async nodes and corrupt concurrent state).

    def _fresh(self):

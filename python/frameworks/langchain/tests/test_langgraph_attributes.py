"""Tests for LangGraph attributes module."""

import pytest

from traceai_langchain._langgraph._attributes import (
    LangGraphAttributes,
    LangGraphSpanKind,
)


class TestLangGraphAttributes:
    """Test LangGraph attribute definitions."""

    def test_graph_structure_attributes_exist(self):
        """Test that graph structure attributes are defined."""
        assert hasattr(LangGraphAttributes, "GRAPH_NAME")
        assert hasattr(LangGraphAttributes, "GRAPH_NODE_COUNT")
        assert hasattr(LangGraphAttributes, "GRAPH_EDGE_COUNT")
        assert hasattr(LangGraphAttributes, "GRAPH_TOPOLOGY")
        assert hasattr(LangGraphAttributes, "GRAPH_ENTRY_POINT")
        assert hasattr(LangGraphAttributes, "GRAPH_CONDITIONAL_EDGE_COUNT")

    def test_node_execution_attributes_exist(self):
        """Test that node execution attributes are defined."""
        assert hasattr(LangGraphAttributes, "NODE_NAME")
        assert hasattr(LangGraphAttributes, "NODE_TYPE")
        assert hasattr(LangGraphAttributes, "NODE_IS_ENTRY")
        assert hasattr(LangGraphAttributes, "NODE_IS_END")

    def test_execution_mode_attributes_exist(self):
        """Test that execution mode attributes are defined."""
        assert hasattr(LangGraphAttributes, "EXECUTION_MODE")
        assert hasattr(LangGraphAttributes, "EXECUTION_SUPERSTEP")
        assert hasattr(LangGraphAttributes, "EXECUTION_THREAD_ID")
        assert hasattr(LangGraphAttributes, "STREAM_MODE")

    def test_state_management_attributes_exist(self):
        """Test that state management attributes are defined."""
        assert hasattr(LangGraphAttributes, "STATE_INPUT")
        assert hasattr(LangGraphAttributes, "STATE_OUTPUT")
        assert hasattr(LangGraphAttributes, "STATE_UPDATES")
        assert hasattr(LangGraphAttributes, "STATE_CHANGED_FIELDS")
        assert hasattr(LangGraphAttributes, "STATE_REDUCER")
        assert hasattr(LangGraphAttributes, "STATE_DIFF")

    def test_checkpoint_attributes_exist(self):
        """Test that checkpoint attributes are defined."""
        assert hasattr(LangGraphAttributes, "CHECKPOINT_THREAD_ID")

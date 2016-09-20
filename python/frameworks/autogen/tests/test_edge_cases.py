"""Edge case tests for AutoGen instrumentation.

These tests verify correct behavior in boundary conditions and unusual situations.
"""

import asyncio
import json
import pytest
from unittest.mock import MagicMock

from traceai_autogen._v04_wrapper import (
    safe_serialize,
    extract_agent_info,
    extract_team_info,
    extract_message_info,
    extract_usage_from_result,
    wrap_agent_on_messages,
    wrap_team_run,
    wrap_tool_execution,
)
from traceai_autogen._attributes import get_model_provider, get_agent_type_name


class TestSafeSerializeEdgeCases:
    """Edge cases for safe_serialize function."""

    def test_serialize_empty_string(self):
        """Test serializing empty string."""
        result = safe_serialize("")
        assert result == ""

    def test_serialize_empty_dict(self):
        """Test serializing empty dict."""
        result = safe_serialize({})
        assert result == "{}"

    def test_serialize_empty_list(self):
        """Test serializing empty list."""
        result = safe_serialize([])
        assert result == "[]"

    def test_serialize_nested_structure(self):
        """Test serializing deeply nested structure."""
        nested = {"level1": {"level2": {"level3": {"data": [1, 2, 3]}}}}
        result = safe_serialize(nested)
        assert "level1" in result
        assert "level3" in result

    def test_serialize_unicode(self):
        """Test serializing unicode characters."""
        result = safe_serialize("Hello 世界 🌍")
        assert "世界" in result

    def test_serialize_special_characters(self):
        """Test serializing special characters."""
        result = safe_serialize("Line1\nLine2\tTabbed")
        assert "\\n" in result or "\n" in result

    def test_serialize_circular_reference_fallback(self):
        """Test serializing object that can't be JSON serialized."""
        obj = MagicMock()
        result = safe_serialize(obj)
        assert isinstance(result, str)

    def test_serialize_bytes(self):
        """Test serializing bytes object."""
        result = safe_serialize(b"hello bytes")
        assert isinstance(result, str)

    def test_serialize_at_max_length(self):
        """Test serializing string exactly at max length."""
        exact_string = "a" * 100
        result = safe_serialize(exact_string, max_length=100)
        assert len(result) == 100
        assert not result.endswith("...")

    def test_serialize_just_over_max_length(self):
        """Test serializing string just over max length."""
        over_string = "a" * 101
        result = safe_serialize(over_string, max_length=100)
        assert len(result) == 100
        assert result.endswith("...")


class TestExtractAgentInfoEdgeCases:
    """Edge cases for extract_agent_info function."""

    def test_agent_with_no_attributes(self):
        """Test agent with minimal attributes."""
        agent = MagicMock(spec=[])
        agent.name = "minimal_agent"
        type(agent).__name__ = "Agent"

        info = extract_agent_info(agent)

        assert info["agent_name"] == "minimal_agent"
        assert info["agent_type"] == "Agent"

    def test_agent_with_empty_name(self):
        """Test agent with empty name."""
        agent = MagicMock()
        agent.name = ""
        type(agent).__name__ = "Agent"

        info = extract_agent_info(agent)

        assert info["agent_name"] == ""

    def test_agent_with_none_model_client(self):
        """Test agent with None model client."""
        agent = MagicMock()
        agent.name = "agent"
        type(agent).__name__ = "Agent"
        agent._model_client = None

        info = extract_agent_info(agent)

        assert "model_name" not in info

    def test_agent_with_empty_tools_list(self):
        """Test agent with empty tools list."""
        agent = MagicMock()
        agent.name = "agent"
        type(agent).__name__ = "Agent"
        agent._tools = []

        info = extract_agent_info(agent)

        assert info.get("tool_count", 0) == 0

    def test_agent_with_many_tools(self):
        """Test agent with many tools."""
        agent = MagicMock()
        agent.name = "agent"
        type(agent).__name__ = "Agent"
        agent._tools = [MagicMock() for _ in range(100)]

        info = extract_agent_info(agent)

        assert info["tool_count"] == 100

    def test_agent_with_very_long_description(self):
        """Test agent with very long description."""
        agent = MagicMock()
        agent.name = "agent"
        type(agent).__name__ = "Agent"
        agent.description = "x" * 1000

        info = extract_agent_info(agent)

        # Should be truncated
        assert len(info["agent_description"]) <= 500


class TestExtractTeamInfoEdgeCases:
    """Edge cases for extract_team_info function."""

    def test_team_with_no_participants(self):
        """Test team with no participants."""
        team = MagicMock()
        type(team).__name__ = "EmptyTeam"
        team._participants = []

        info = extract_team_info(team)

        assert info["team_type"] == "EmptyTeam"
        assert "participant_count" not in info

    def test_team_with_single_participant(self):
        """Test team with single participant."""
        agent = MagicMock()
        agent.name = "solo_agent"

        team = MagicMock()
        type(team).__name__ = "Team"
        team._participants = [agent]

        info = extract_team_info(team)

        assert info["participant_count"] == 1
        assert info["participants"] == ["solo_agent"]

    def test_team_with_many_participants(self):
        """Test team with many participants."""
        agents = [MagicMock(name=f"agent_{i}") for i in range(50)]
        for i, a in enumerate(agents):
            a.name = f"agent_{i}"

        team = MagicMock()
        type(team).__name__ = "LargeTeam"
        team._participants = agents

        info = extract_team_info(team)

        assert info["participant_count"] == 50

    def test_team_with_none_termination(self):
        """Test team with None termination condition."""
        team = MagicMock()
        type(team).__name__ = "Team"
        team._participants = []
        team._termination_condition = None

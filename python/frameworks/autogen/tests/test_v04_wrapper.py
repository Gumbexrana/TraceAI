"""Tests for AutoGen v0.4 wrapper functions."""

import asyncio
import json
import pytest
import time
from unittest.mock import MagicMock, AsyncMock, patch

from traceai_autogen._v04_wrapper import (
    safe_serialize,
    extract_agent_info,
    extract_team_info,
    extract_message_info,
    extract_usage_from_result,
    wrap_agent_on_messages,
    wrap_team_run,
    wrap_tool_execution,
    TracedTeamRunStream,
)


class TestSafeSerialize:
    """Tests for safe_serialize function."""

    def test_serialize_none(self):
        """Test serializing None."""
        result = safe_serialize(None)
        assert result == ""

    def test_serialize_string(self):
        """Test serializing string."""
        result = safe_serialize("hello world")
        assert result == "hello world"

    def test_serialize_dict(self):
        """Test serializing dictionary."""
        result = safe_serialize({"key": "value"})
        assert result == '{"key": "value"}'

    def test_serialize_list(self):
        """Test serializing list."""
        result = safe_serialize([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_serialize_truncates_long_strings(self):
        """Test that long strings are truncated."""
        long_string = "a" * 3000
        result = safe_serialize(long_string, max_length=100)
        assert len(result) == 100
        assert result.endswith("...")

    def test_serialize_object(self):
        """Test serializing arbitrary object."""
        obj = MagicMock()
        obj.__str__ = MagicMock(return_value="MockObject")
        result = safe_serialize(obj)
        assert "MockObject" in result or "MagicMock" in result


class TestExtractAgentInfo:
    """Tests for extract_agent_info function."""

    def test_extract_basic_info(self):
        """Test extracting basic agent info."""
        agent = MagicMock()
        agent.name = "test_agent"
        type(agent).__name__ = "AssistantAgent"
        agent.description = "A test agent"
        agent._model_client = None
        agent._tools = []
        agent._memory = None

        info = extract_agent_info(agent)

        assert info["agent_name"] == "test_agent"
        assert info["agent_type"] == "AssistantAgent"
        assert info.get("agent_description") == "A test agent"

    def test_extract_with_model_client(self):
        """Test extracting info with model client."""
        agent = MagicMock()
        agent.name = "assistant"
        type(agent).__name__ = "AssistantAgent"
        agent.description = "Helper"
        agent._model_client = MagicMock()
        agent._model_client.model = "gpt-4"
        agent._tools = [MagicMock(), MagicMock()]
        agent._memory = True

        info = extract_agent_info(agent)

        assert info["agent_name"] == "assistant"
        assert info["model_name"] == "gpt-4"
        assert info["tool_count"] == 2
        assert info["has_memory"] is True

    def test_extract_with_private_model(self):
        """Test extracting model from _model attribute."""
        agent = MagicMock()
        agent.name = "agent"
        type(agent).__name__ = "Agent"
        del agent.description
        agent._model_client = MagicMock(spec=[])
        agent._model_client._model = "claude-3"
        del agent._tools
        del agent._memory

        info = extract_agent_info(agent)

        assert info["model_name"] == "claude-3"


class TestExtractTeamInfo:
    """Tests for extract_team_info function."""

    def test_extract_basic_team_info(self):
        """Test extracting basic team info."""
        team = MagicMock()
        type(team).__name__ = "RoundRobinGroupChat"
        team._participants = []
        team._termination_condition = None
        team._max_turns = None

        info = extract_team_info(team)

        assert info["team_type"] == "RoundRobinGroupChat"

    def test_extract_team_with_participants(self):
        """Test extracting team with participants."""
        agent1 = MagicMock()
        agent1.name = "agent1"
        agent2 = MagicMock()
        agent2.name = "agent2"

        team = MagicMock()
        type(team).__name__ = "SelectorGroupChat"
        team._participants = [agent1, agent2]
        team._termination_condition = MagicMock()
        type(team._termination_condition).__name__ = "MaxMessageTermination"
        team._max_turns = 10

        info = extract_team_info(team)

        assert info["team_type"] == "SelectorGroupChat"
        assert info["participant_count"] == 2
        assert info["participants"] == ["agent1", "agent2"]
        assert info["termination_condition"] == "MaxMessageTermination"
        assert info["max_turns"] == 10


class TestExtractMessageInfo:
    """Tests for extract_message_info function."""

    def test_extract_text_message(self):
        """Test extracting text message info."""
        message = MagicMock()
        type(message).__name__ = "TextMessage"
        message.content = "Hello, world!"
        message.source = "user"
        message.models_usage = None

        info = extract_message_info(message)

        assert info["message_type"] == "TextMessage"
        assert info["content"] == "Hello, world!"
        assert info["source"] == "user"

    def test_extract_message_with_usage(self):
        """Test extracting message with token usage."""
        message = MagicMock()
        type(message).__name__ = "AssistantMessage"
        message.content = "Response text"
        message.source = "assistant"
        message.models_usage = {"gpt-4": {"prompt_tokens": 100, "completion_tokens": 50}}

        info = extract_message_info(message)

        assert info["message_type"] == "AssistantMessage"
        assert "models_usage" in info


class TestExtractUsageFromResult:
    """Tests for extract_usage_from_result function."""

    def test_extract_no_result(self):
        """Test with no result."""
        assert extract_usage_from_result(None) is None

    def test_extract_no_messages(self):
        """Test with result but no messages."""
        result = MagicMock()
        result.messages = None
        assert extract_usage_from_result(result) is None


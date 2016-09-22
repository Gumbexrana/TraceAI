"""Real-world scenario tests for AutoGen instrumentation.

These tests verify the instrumentation works correctly in realistic use cases
without requiring actual AutoGen or API calls.
"""

import asyncio
import json
import time
import pytest
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
)
from traceai_autogen._attributes import (
    AutoGenSpanKind,
    AutoGenAttributes,
    get_model_provider,
)


class TestChatbotScenario:
    """Test scenarios for a simple chatbot."""

    def test_single_turn_conversation(self):
        """Test a single question-answer exchange."""
        tracer = MagicMock()
        span = MagicMock()
        span.__enter__ = MagicMock(return_value=span)
        span.__exit__ = MagicMock(return_value=None)
        tracer.start_as_current_span = MagicMock(return_value=span)

        async def chatbot_response(self, messages, cancellation_token=None):
            # Simulate processing time
            response = MagicMock()
            response.chat_message = MagicMock()
            response.chat_message.content = "Paris is the capital of France."
            return response

        wrapped = wrap_agent_on_messages(chatbot_response, tracer)

        agent = MagicMock()
        agent.name = "chatbot"
        type(agent).__name__ = "AssistantAgent"
        agent._model_client = MagicMock()
        agent._model_client.model = "gpt-4o"
        agent._tools = []

        message = MagicMock()
        message.content = "What is the capital of France?"
        message.source = "user"

        async def run_test():
            result = await wrapped(agent, [message])
            return result

        result = asyncio.run(run_test())

        assert result.chat_message.content == "Paris is the capital of France."
        tracer.start_as_current_span.assert_called_once()

    def test_multi_turn_conversation(self):
        """Test a multi-turn conversation with history."""
        tracer = MagicMock()
        span = MagicMock()
        span.__enter__ = MagicMock(return_value=span)
        span.__exit__ = MagicMock(return_value=None)
        tracer.start_as_current_span = MagicMock(return_value=span)

        conversation_state = {"turn": 0}

        async def chatbot_response(self, messages, cancellation_token=None):
            conversation_state["turn"] += 1
            response = MagicMock()
            response.chat_message = MagicMock()
            response.chat_message.content = f"Response for turn {conversation_state['turn']}"
            return response

        wrapped = wrap_agent_on_messages(chatbot_response, tracer)

        agent = MagicMock()
        agent.name = "chatbot"
        type(agent).__name__ = "AssistantAgent"
        agent._model_client = None
        agent._tools = []

        async def run_test():
            # Turn 1
            msg1 = MagicMock(content="Hello", source="user")
            result1 = await wrapped(agent, [msg1])

            # Turn 2 (with history)
            msg2 = MagicMock(content="How are you?", source="user")
            result2 = await wrapped(agent, [msg1, result1.chat_message, msg2])

            # Turn 3
            msg3 = MagicMock(content="Goodbye", source="user")
            result3 = await wrapped(agent, [msg1, result1.chat_message, msg2, result2.chat_message, msg3])

            return [result1, result2, result3]

        results = asyncio.run(run_test())

        assert len(results) == 3
        assert conversation_state["turn"] == 3
        assert tracer.start_as_current_span.call_count == 3


class TestTeamCollaborationScenario:
    """Test scenarios for multi-agent team collaboration."""

    def test_round_robin_team(self):
        """Test round-robin team collaboration."""
        tracer = MagicMock()
        span = MagicMock()
        span.__enter__ = MagicMock(return_value=span)
        span.__exit__ = MagicMock(return_value=None)
        tracer.start_as_current_span = MagicMock(return_value=span)

        async def team_run(self, task, *args, **kwargs):
            # Simulate team discussion
            result = MagicMock()
            result.messages = [
                MagicMock(content="I'll start with the architecture.", source="architect"),
                MagicMock(content="Good plan. I'll implement it.", source="developer"),
                MagicMock(content="I'll write tests for it.", source="tester"),
            ]
            result.stop_reason = "max_messages"
            return result

        wrapped = wrap_team_run(team_run, tracer, "run")

        # Create proper participant mocks with string names
        architect = MagicMock()
        architect.name = "architect"
        developer = MagicMock()
        developer.name = "developer"
        tester = MagicMock()
        tester.name = "tester"

        team = MagicMock()
        type(team).__name__ = "RoundRobinGroupChat"
        team._participants = [architect, developer, tester]
        team._termination_condition = MagicMock()
        type(team._termination_condition).__name__ = "MaxMessageTermination"
        team._max_turns = 10

        async def run_test():
            result = await wrapped(team, "Build a REST API")
            return result

        result = asyncio.run(run_test())

        assert len(result.messages) == 3
        assert result.stop_reason == "max_messages"
        tracer.start_as_current_span.assert_called_once()

    def test_selector_team_routing(self):
        """Test selector-based team routing."""
        tracer = MagicMock()
        span = MagicMock()
        span.__enter__ = MagicMock(return_value=span)
        span.__exit__ = MagicMock(return_value=None)
        tracer.start_as_current_span = MagicMock(return_value=span)

        async def team_run(self, task, *args, **kwargs):
            # Selector chose the right agent
            result = MagicMock()
            result.messages = [
                MagicMock(content="This is a frontend question. Let me handle it.", source="frontend_dev"),
                MagicMock(content="Here's the React component you need...", source="frontend_dev"),
            ]
            result.stop_reason = "TextMentionTermination"
            return result

        wrapped = wrap_team_run(team_run, tracer, "run")

        # Create proper participant mocks with string names
        backend = MagicMock()
        backend.name = "backend_dev"
        frontend = MagicMock()
        frontend.name = "frontend_dev"
        devops = MagicMock()
        devops.name = "devops"

        team = MagicMock()
        type(team).__name__ = "SelectorGroupChat"
        team._participants = [backend, frontend, devops]
        team._termination_condition = None
        team._max_turns = None

        async def run_test():
            result = await wrapped(team, "Create a React component for user profile")
            return result

        result = asyncio.run(run_test())

        # Frontend dev handled it
        assert all(msg.source == "frontend_dev" for msg in result.messages)


class TestToolUsageScenario:
    """Test scenarios for agent tool usage."""

    def test_weather_tool(self):
        """Test weather lookup tool."""
        tracer = MagicMock()
        span = MagicMock()
        span.__enter__ = MagicMock(return_value=span)
        span.__exit__ = MagicMock(return_value=None)
        tracer.start_as_current_span = MagicMock(return_value=span)

        def get_weather(city: str) -> str:
            weather_data = {"new york": "72°F, sunny", "london": "58°F, cloudy"}
            return weather_data.get(city.lower(), "Unknown city")

        wrapped = wrap_tool_execution(get_weather, tracer, "get_weather", "Get weather for a city")

        result = wrapped("New York")

        assert result == "72°F, sunny"
        tracer.start_as_current_span.assert_called_once()

    def test_database_query_tool(self):
        """Test database query tool."""
        tracer = MagicMock()
        span = MagicMock()
        span.__enter__ = MagicMock(return_value=span)
        span.__exit__ = MagicMock(return_value=None)
        tracer.start_as_current_span = MagicMock(return_value=span)


"""Pytest fixtures for traceai-agno tests."""

import pytest
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List, Optional


class MockModel:
    """Mock Agno model."""

    def __init__(
        self,
        model_id: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 1.0,
    ):
        self.id = model_id
        self.model = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p


class MockTool:
    """Mock Agno tool."""

    def __init__(
        self,
        name: str = "test_tool",
        description: str = "A test tool",
    ):
        self.name = name
        self.description = description

    def __call__(self, *args, **kwargs):
        return "Tool executed"


class MockAgent:
    """Mock Agno Agent."""

    def __init__(
        self,
        name: str = "TestAgent",
        agent_id: str = "agent-123",
        description: str = "A test agent",
        instructions: str = "Be helpful and accurate",
        model: Optional[MockModel] = None,
        tools: Optional[List[Any]] = None,
        debug_mode: bool = False,
        markdown: bool = True,
        memory: Optional[Any] = None,
        knowledge: Optional[Any] = None,
    ):
        self.name = name
        self.agent_id = agent_id
        self.description = description
        self.instructions = instructions
        self.model = model or MockModel()
        self.tools = tools or []
        self.debug_mode = debug_mode
        self.markdown = markdown
        self.memory = memory
        self.knowledge = knowledge

    def run(self, message: str) -> str:
        return f"Agent response to: {message}"

    async def arun(self, message: str) -> str:
        return f"Async agent response to: {message}"


class MockTeam:
    """Mock Agno Team."""

    def __init__(
        self,
        name: str = "TestTeam",
        agents: Optional[List[MockAgent]] = None,
    ):
        self.name = name
        self.agents = agents or []

    def run(self, message: str) -> str:
        return f"Team response to: {message}"


class MockWorkflow:
    """Mock Agno Workflow."""

    def __init__(self, name: str = "TestWorkflow"):
        self.name = name

    def run(self, *args, **kwargs):
        return "Workflow result"


class MockUsage:
    """Mock token usage."""

    def __init__(
        self,
        input_tokens: int = 100,
        output_tokens: int = 50,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.prompt_tokens = input_tokens
        self.completion_tokens = output_tokens


class MockResponse:
    """Mock LLM response."""

    def __init__(
        self,

"""AutoGen v0.4 agent with tools example.

This example demonstrates:
- Defining tool functions for the agent
- Agent using tools to accomplish tasks
- Tracing tool calls with inputs and outputs
"""

import asyncio
import os
import json
from datetime import datetime
from typing import Annotated

from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from traceai_autogen import instrument_autogen

# Setup tracing
trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name="autogen-agent-tools",
)
instrument_autogen(tracer_provider=trace_provider)

# Import AutoGen v0.4
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient


# Define tools for the agent
def get_weather(
    city: Annotated[str, "The city name to get weather for"],
) -> str:
    """Get the current weather for a city."""
    # Simulated weather data
    weather_data = {
        "new york": {"temp": 72, "condition": "sunny", "humidity": 45},
        "london": {"temp": 58, "condition": "cloudy", "humidity": 78},
        "tokyo": {"temp": 68, "condition": "partly cloudy", "humidity": 60},
        "paris": {"temp": 65, "condition": "rainy", "humidity": 82},
    }

    city_lower = city.lower()
    if city_lower in weather_data:
        w = weather_data[city_lower]
        return f"Weather in {city}: {w['temp']}°F, {w['condition']}, {w['humidity']}% humidity"
    else:
        return f"Weather data not available for {city}"


def calculate(
    expression: Annotated[str, "Mathematical expression to evaluate"],
) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        # Safe evaluation of math expressions
        allowed_chars = set("0123456789+-*/.(). ")

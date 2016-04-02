"""Agno agent with tools example with TraceAI instrumentation.

This example demonstrates how to set up tracing for an agent with tool calling.
"""

import os
from typing import Optional

# Setup tracing BEFORE importing Agno
from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from traceai_agno import configure_agno_tracing

# Initialize TraceAI
trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name="agno-tools-example",
)

# Configure Agno to use TraceAI
configure_agno_tracing(tracer_provider=trace_provider)

# Now import Agno modules
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool


# Define tools
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city to get weather for.

    Returns:
        A string describing the current weather.
    """
    # Mock weather data
    weather_data = {
        "new york": "Sunny, 72°F (22°C)",
        "london": "Cloudy, 59°F (15°C)",
        "tokyo": "Rainy, 68°F (20°C)",
        "paris": "Partly cloudy, 65°F (18°C)",
        "sydney": "Clear, 77°F (25°C)",
    }
    city_lower = city.lower()
    return weather_data.get(city_lower, f"Weather data not available for {city}")


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

# TraceAI Strands Agents Integration

Comprehensive observability for [AWS Strands Agents](https://strandsagents.com/) with TraceAI.

Strands Agents is an open-source SDK from AWS that enables building AI agents with a model-driven approach. This integration provides seamless tracing by leveraging Strands' built-in OpenTelemetry support.

## Features

- **Zero-config integration**: Works with Strands' native OTEL support
- **Automatic tracing**: Agent invocations, tool calls, and model interactions
- **Token tracking**: Input/output tokens and cache metrics (Bedrock)
- **Session correlation**: Link traces across conversations
- **Custom callbacks**: Extended event capture for detailed observability
- **MCP support**: Trace Model Context Protocol tool usage

## Installation

```bash
pip install traceai-strands
```

For full functionality with Strands:

```bash
pip install 'strands-agents[otel]'
```

## Quick Start

### Option 1: Using TraceAI with fi_instrumentation

```python
from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from traceai_strands import configure_strands_tracing

# Setup TraceAI
trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name="my-strands-agent",
)

# Configure Strands to use TraceAI
configure_strands_tracing(tracer_provider=trace_provider)

# Now use Strands normally - traces are sent automatically
from strands import Agent

agent = Agent(
    model="us.anthropic.claude-sonnet-4-20250514-v1:0",
    system_prompt="You are a helpful assistant.",
)

response = agent("Hello!")
```

### Option 2: Direct OTLP Configuration

```python
from traceai_strands import configure_strands_tracing

# Configure with OTLP endpoint directly
configure_strands_tracing(
    otlp_endpoint="https://api.traceai.com/v1/traces",
    otlp_headers={"Authorization": "Bearer YOUR_API_KEY"},
    project_name="my-strands-agent",
)

from strands import Agent

agent = Agent(
    model="us.anthropic.claude-sonnet-4-20250514-v1:0",
    system_prompt="You are a helpful assistant.",
)

response = agent("Hello!")
```

## Adding Trace Attributes

Add session and user information for better trace correlation:

```python
from strands import Agent

agent = Agent(
    model="us.anthropic.claude-sonnet-4-20250514-v1:0",
    system_prompt="You are a helpful assistant.",
    trace_attributes={
        "session.id": "user-session-123",
        "user.id": "user@example.com",
        "tags": ["production", "chatbot"],
    },
)
```

Or use the helper function:

```python
from traceai_strands import create_traced_agent

agent = create_traced_agent(
    model="us.anthropic.claude-sonnet-4-20250514-v1:0",
    system_prompt="You are a helpful assistant.",
    session_id="user-session-123",
    user_id="user@example.com",
    tags=["production", "chatbot"],
)
```

## Using Tools

Strands tools are automatically traced when using the `@tool` decorator:

```python
from strands import Agent, tool
from typing import Annotated

@tool
def get_weather(city: Annotated[str, "City name"]) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: 72°F, Sunny"

@tool
def calculate(
    operation: Annotated[str, "add, subtract, multiply, divide"],
    a: Annotated[float, "First number"],
    b: Annotated[float, "Second number"],
) -> float:
    """Perform a calculation."""
    ops = {"add": lambda x, y: x + y, "multiply": lambda x, y: x * y}
    return ops[operation](a, b)

agent = Agent(
    model="us.anthropic.claude-sonnet-4-20250514-v1:0",
    tools=[get_weather, calculate],
)

response = agent("What's 15 times 7, and what's the weather in Tokyo?")
```

## Custom Callback Handler

For extended event capture beyond Strands' built-in telemetry:

```python
from traceai_strands import StrandsCallbackHandler

# Create callback handler
callback = StrandsCallbackHandler(
    tracer_provider=trace_provider,
    capture_input=True,
    capture_output=True,
)

# Use with agent
agent = Agent(
    model="us.anthropic.claude-sonnet-4-20250514-v1:0",
    callback_handler=callback,
)

# Lifecycle events are automatically traced
response = agent("Hello!")
```

## MCP Integration

Trace Model Context Protocol server tools:

```python
from strands import Agent
from strands.tools.mcp import MCPClient

# Connect to MCP server
mcp_client = MCPClient(
    server_command=["npx", "@anthropic/mcp-server-calculator"],
)

# Get MCP tools
mcp_tools = mcp_client.list_tools_sync()

# Create agent with MCP tools
agent = Agent(
    model="us.anthropic.claude-sonnet-4-20250514-v1:0",
    tools=mcp_tools,
)

response = agent("Calculate the square root of 144")
```

## Semantic Attributes

The integration captures these OpenTelemetry GenAI semantic attributes:

### Agent Attributes
| Attribute | Description |
|-----------|-------------|
| `agent.type` | Agent class name |
| `strands.system_prompt` | Agent's system prompt (truncated) |
| `strands.tool_count` | Number of tools available |
| `strands.session.id` | Session identifier |

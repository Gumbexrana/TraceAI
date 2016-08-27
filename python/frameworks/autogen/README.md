# TraceAI AutoGen Instrumentation

OpenTelemetry instrumentation for [Microsoft AutoGen](https://microsoft.github.io/autogen/), providing comprehensive tracing for multi-agent conversations, tool executions, and LLM interactions.

**Supports both AutoGen v0.2 (legacy) and v0.4 (AgentChat).**

## Installation

```bash
pip install traceAI-autogen
```

### AutoGen Versions

This package supports both major AutoGen versions:

**AutoGen v0.2 (legacy)**:
```bash
pip install autogen>=0.2.0
```

**AutoGen v0.4 (AgentChat)**:
```bash
pip install autogen-agentchat>=0.4.0
```

## Quick Start

### Set Environment Variables

```python
import os

os.environ["FI_API_KEY"] = "your-api-key"
os.environ["FI_SECRET_KEY"] = "your-secret-key"
os.environ["OPENAI_API_KEY"] = "your-openai-key"
```

### Register Tracer Provider

```python
from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType

trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name="autogen_app"
)
```

### Instrument AutoGen

```python
from traceai_autogen import AutogenInstrumentor

AutogenInstrumentor().instrument(tracer_provider=trace_provider)
```

Or use the convenience function:

```python
from traceai_autogen import instrument_autogen

instrumentor = instrument_autogen(tracer_provider=trace_provider)
```

## Examples

### AutoGen v0.2 (Legacy)

```python
import autogen
from traceai_autogen import instrument_autogen

# Instrument AutoGen
instrument_autogen()

# Configure LLM
llm_config = {
    "config_list": [{"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}],
    "temperature": 0,
}

# Create agents
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="You are a helpful AI assistant."
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    code_execution_config={"work_dir": "coding", "use_docker": False},
)

# Start conversation - automatically traced
chat_result = user_proxy.initiate_chat(
    assistant,
    message="Write a Python function to calculate fibonacci numbers."
)
```

### AutoGen v0.4 (AgentChat)

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from traceai_autogen import instrument_autogen

# Instrument AutoGen
instrument_autogen()

async def main():
    # Create model client
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    # Create agents
    coder = AssistantAgent(
        name="coder",
        model_client=model_client,
        system_message="You are a Python expert. Write clean, efficient code.",
    )

    reviewer = AssistantAgent(
        name="reviewer",
        model_client=model_client,
        system_message="You review code and suggest improvements.",
    )

    # Create team
    team = RoundRobinGroupChat(
        participants=[coder, reviewer],
        termination_condition=MaxMessageTermination(max_messages=6),
    )

    # Run team task - automatically traced
    result = await team.run(task="Write a Python class for a binary search tree.")

    print(result.messages[-1].content)

asyncio.run(main())
```

### AutoGen v0.4 with Tools

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from traceai_autogen import instrument_autogen

instrument_autogen()

# Define tools
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny and 72F."

def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"


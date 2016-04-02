# TraceAI Agno Instrumentation

OpenTelemetry instrumentation for [Agno](https://github.com/agno-agi/agno), the high-performance AI agent framework.

## Installation

```bash
pip install traceai-agno
```

For full functionality, also install the Agno framework and its OpenInference instrumentation:

```bash
pip install agno openinference-instrumentation-agno
```

## Quick Start

```python
from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from traceai_agno import configure_agno_tracing

# Setup TraceAI
trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name="agno-agent",
)

# Configure Agno to use TraceAI (call BEFORE creating agents)
configure_agno_tracing(tracer_provider=trace_provider)

# Now import and use Agno normally
from agno.agent import Agent
from agno.models.openai import OpenAIChat

agent = Agent(
    model=OpenAIChat(id="gpt-4"),
    description="A helpful assistant",
)

response = agent.run("What is the capital of France?")
print(response.content)
```

## Configuration Options

### Using fi_instrumentation (Recommended)

```python
from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from traceai_agno import configure_agno_tracing

trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name="my-agno-project",
)

configure_agno_tracing(tracer_provider=trace_provider)
```

### Direct OTLP Configuration

```python
from traceai_agno import configure_agno_tracing

configure_agno_tracing(
    otlp_endpoint="https://api.traceai.com/v1/traces",
    otlp_headers={"Authorization": "Bearer YOUR_API_KEY"},
    project_name="my-agno-project",
)
```

### Using Custom Tracer Provider

```python
from traceai_agno import setup_traceai_exporter, configure_agno_tracing

# Create a custom tracer provider
provider = setup_traceai_exporter(
    endpoint="https://api.traceai.com/v1/traces",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    service_name="my-agno-service",
    use_batch_processor=True,
)

configure_agno_tracing(tracer_provider=provider)
```

## Features


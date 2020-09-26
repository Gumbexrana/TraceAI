# TraceAI Pydantic AI Instrumentation

OpenTelemetry instrumentation for [Pydantic AI](https://ai.pydantic.dev), providing comprehensive tracing for AI agent executions, tool calls, and model interactions.

## Installation

```bash
pip install traceai-pydantic-ai
```

## Quick Start

```python
from traceai_pydantic_ai import PydanticAIInstrumentor

# Initialize instrumentation (do this once at startup)
PydanticAIInstrumentor().instrument()

# Use Pydantic AI as normal - tracing is automatic
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', instructions='Be concise and helpful.')
result = agent.run_sync('What is the capital of France?')
print(result.output)
```

## Features

- **Agent Run Tracing**: Automatic tracing for `run()`, `run_sync()`, and `run_stream()` methods
- **Tool Call Tracking**: Traces tool function executions with inputs and outputs
- **Model Attribution**: Automatic detection of model provider (OpenAI, Anthropic, Google, etc.)
- **Usage Metrics**: Token usage tracking (input, output, total)
- **Streaming Support**: Full tracing for streaming responses with chunk counting
- **Error Handling**: Comprehensive error recording and exception tracking
- **Structured Outputs**: Support for Pydantic model result types
- **Conversation Memory**: Message history length tracking
- **Multi-model Support**: Traces across different model providers

## Configuration

### Basic Setup

```python
from traceai_pydantic_ai import PydanticAIInstrumentor

# Use default tracer provider
PydanticAIInstrumentor().instrument()
```

### With Custom Tracer Provider

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from traceai_pydantic_ai import PydanticAIInstrumentor

# Setup tracer provider
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# Instrument with provider
PydanticAIInstrumentor().instrument(tracer_provider=provider)
```

### Convenience Function

```python
from traceai_pydantic_ai import instrument_pydantic_ai

# One-liner instrumentation
instrumentor = instrument_pydantic_ai()
```

### Using Built-in Instrumentation

Pydantic AI has its own OpenTelemetry support. You can optionally use that instead:

```python
from traceai_pydantic_ai import instrument_pydantic_ai

# Use Pydantic AI's built-in OTEL (via Agent.instrument_all())
instrument_pydantic_ai(use_builtin=True)
```

## Examples

### Basic Agent

```python
from traceai_pydantic_ai import PydanticAIInstrumentor
from pydantic_ai import Agent

# Initialize instrumentation
PydanticAIInstrumentor().instrument()

# Create agent
agent = Agent(
    'openai:gpt-4o',
    instructions='You are a helpful assistant.'
)

# Synchronous execution
result = agent.run_sync('Hello, how are you?')
print(result.output)
```

### Async Agent

```python
import asyncio
from traceai_pydantic_ai import PydanticAIInstrumentor
from pydantic_ai import Agent

PydanticAIInstrumentor().instrument()

agent = Agent('anthropic:claude-3-sonnet')

async def main():
    result = await agent.run('Tell me a joke')
    print(result.output)

asyncio.run(main())
```

### Agent with Tools

```python
from traceai_pydantic_ai import PydanticAIInstrumentor
from pydantic_ai import Agent, RunContext

PydanticAIInstrumentor().instrument()

agent = Agent('openai:gpt-4o')

@agent.tool
def get_weather(ctx: RunContext, city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny and 72°F"

@agent.tool
def calculate(ctx: RunContext, expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

result = agent.run_sync('What is the weather in San Francisco?')
print(result.output)
```

### Structured Outputs

```python
from pydantic import BaseModel
from traceai_pydantic_ai import PydanticAIInstrumentor
from pydantic_ai import Agent

PydanticAIInstrumentor().instrument()

class CityInfo(BaseModel):
    name: str
    country: str
    population: int

agent = Agent('openai:gpt-4o', result_type=CityInfo)
result = agent.run_sync('Tell me about Paris')

city = result.output
print(f"{city.name}, {city.country} - Population: {city.population:,}")
```

### Streaming

```python
import asyncio
from traceai_pydantic_ai import PydanticAIInstrumentor
from pydantic_ai import Agent


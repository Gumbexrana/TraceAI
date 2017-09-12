# TraceAI DeepSeek Instrumentation

OpenTelemetry instrumentation for [DeepSeek](https://www.deepseek.com/) - chat completions, reasoning models, and function calling.

## Installation

```bash
pip install traceai-deepseek
```

## Features

- Automatic tracing of DeepSeek API calls via OpenAI SDK
- Support for DeepSeek Chat, Coder, and Reasoner (R1) models
- Streaming response support
- Token usage tracking including prompt cache metrics
- Reasoning content extraction for R1 models
- Function/tool calling support
- Full OpenTelemetry semantic conventions compliance

## Usage

### Basic Setup

```python
from openai import OpenAI
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from traceai_deepseek import DeepSeekInstrumentor

# Set up tracing
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# Instrument DeepSeek
DeepSeekInstrumentor().instrument(tracer_provider=provider)

# Use DeepSeek via OpenAI SDK
client = OpenAI(
    api_key="your-deepseek-api-key",
    base_url="https://api.deepseek.com/v1"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### DeepSeek Chat

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-deepseek-api-key",
    base_url="https://api.deepseek.com/v1"
)

# Simple chat
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ],
    temperature=0.7,
    max_tokens=1024
)
print(response.choices[0].message.content)
```

### DeepSeek Reasoner (R1 Models)

DeepSeek R1 models provide reasoning capabilities with detailed thought processes.

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-deepseek-api-key",
    base_url="https://api.deepseek.com/v1"
)

# Using DeepSeek Reasoner
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "user", "content": "Solve this step by step: If a train travels 120 km in 2 hours, what is its average speed?"}
    ]
)

# Access the reasoning content (thought process)
message = response.choices[0].message
print("Reasoning:", message.reasoning_content)
print("Answer:", message.content)

# Token usage includes reasoning tokens
print(f"Reasoning tokens: {response.usage.completion_tokens_details.reasoning_tokens}")
```

### Streaming Responses

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-deepseek-api-key",
    base_url="https://api.deepseek.com/v1"
)

# Streaming chat
stream = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()

# Streaming with R1 (includes reasoning)
stream = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[{"role": "user", "content": "What is 15% of 240?"}],
    stream=True
)

reasoning_content = ""
content = ""
for chunk in stream:
    delta = chunk.choices[0].delta
    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
        reasoning_content += delta.reasoning_content
    if delta.content:
        content += delta.content

print("Reasoning:", reasoning_content)
print("Answer:", content)
```

### Function Calling / Tools

```python
from openai import OpenAI
import json

client = OpenAI(
    api_key="your-deepseek-api-key",
    base_url="https://api.deepseek.com/v1"
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto"
)

message = response.choices[0].message
if message.tool_calls:
    for tool_call in message.tool_calls:
        print(f"Function: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")
```

### Async Usage

```python
import asyncio
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(
        api_key="your-deepseek-api-key",
        base_url="https://api.deepseek.com/v1"
    )

    response = await client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

### JSON Mode

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-deepseek-api-key",
    base_url="https://api.deepseek.com/v1"
)

response = client.chat.completions.create(
    model="deepseek-chat",

# TraceAI Together Instrumentation

OpenTelemetry instrumentation for [Together AI](https://together.ai/) - chat completions, completions, and embeddings APIs.

## Installation

```bash
pip install traceai-together
```

## Features

- Automatic tracing of Together AI API calls
- Support for chat completions, completions, and embeddings endpoints
- Streaming response support for both sync and async clients
- Token usage tracking
- Tool/function calling support
- Full OpenTelemetry semantic conventions compliance

## Usage

### Basic Setup

```python
import together
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from traceai_together import TogetherInstrumentor

# Set up tracing
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# Instrument Together AI
TogetherInstrumentor().instrument(tracer_provider=provider)

# Use Together AI
client = together.Together(api_key="your-api-key")
response = client.chat.completions.create(
    model="meta-llama/Llama-3-8b-chat-hf",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Chat Completions

```python
import together

client = together.Together()

# Simple chat
response = client.chat.completions.create(
    model="meta-llama/Llama-3-8b-chat-hf",
    messages=[{"role": "user", "content": "What is machine learning?"}],
    max_tokens=512,
)
print(response.choices[0].message.content)

# With system message
response = client.chat.completions.create(
    model="meta-llama/Llama-3-8b-chat-hf",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing."},
    ],
    temperature=0.7,
)
```

### Streaming Chat

```python
import together

client = together.Together()

# Streaming response
stream = client.chat.completions.create(
    model="meta-llama/Llama-3-8b-chat-hf",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Completions (Legacy)

```python
import together

client = together.Together()

# Text completion
response = client.completions.create(
    model="meta-llama/Llama-3-8b-hf",
    prompt="The quick brown fox",
    max_tokens=50,
)
print(response.choices[0].text)
```

### Embeddings

```python
import together

client = together.Together()

# Generate embeddings
response = client.embeddings.create(
    model="togethercomputer/m2-bert-80M-8k-retrieval",
    input=["Hello world", "Machine learning is great"],
)
print(f"Generated {len(response.data)} embeddings")
print(f"Dimensions: {len(response.data[0].embedding)}")
```

### Async Client

```python
import asyncio
import together

async def main():

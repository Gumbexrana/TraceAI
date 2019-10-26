# TraceAI Ollama Instrumentation

OpenTelemetry instrumentation for [Ollama](https://ollama.ai/) - the leading local LLM runner.

## Installation

```bash
pip install traceai-ollama
```

## Features

- Automatic tracing of Ollama API calls
- Support for chat, generate, and embed endpoints
- Streaming response support
- Token usage tracking
- Performance metrics (total_duration, eval_duration, etc.)
- Full OpenTelemetry semantic conventions compliance

## Usage

### Basic Setup

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from traceai_ollama import OllamaInstrumentor
import ollama

# Set up tracing
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# Instrument Ollama
OllamaInstrumentor().instrument(tracer_provider=provider)

# Use Ollama - calls are automatically traced
response = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response["message"]["content"])
```

### Chat Completions

```python
import ollama

# Simple chat
response = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

# Multi-turn conversation
messages = [
    {"role": "user", "content": "My name is Alice."},
    {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
    {"role": "user", "content": "What's my name?"}
]
response = ollama.chat(model="llama3.2", messages=messages)
```

### Streaming Responses

```python
import ollama

# Streaming chat
stream = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Tell me a story."}],
    stream=True
)

for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)
```

### Text Generation

```python
import ollama

# Simple text generation
response = ollama.generate(
    model="llama3.2",
    prompt="The quick brown fox"
)
print(response["response"])

# With system prompt
response = ollama.generate(
    model="llama3.2",
    prompt="Write a haiku",
    system="You are a poet."
)
```

### Embeddings

```python
import ollama

# Single embedding
response = ollama.embed(
    model="nomic-embed-text",
    input="Hello, world!"

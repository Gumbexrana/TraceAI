# TraceAI HuggingFace Instrumentation

OpenTelemetry instrumentation for [HuggingFace Inference API](https://huggingface.co/docs/huggingface_hub/guides/inference) - text generation, chat completion, and embeddings.

## Installation

```bash
pip install traceai-huggingface
```

## Features

- Automatic tracing of HuggingFace Inference API calls
- Support for text generation, chat completion, and feature extraction (embeddings)
- Streaming response support
- Token usage tracking
- Both sync and async client support
- Full OpenTelemetry semantic conventions compliance

## Usage

### Basic Setup

```python
from huggingface_hub import InferenceClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from traceai_huggingface import HuggingFaceInstrumentor

# Set up tracing
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# Instrument HuggingFace
HuggingFaceInstrumentor().instrument(tracer_provider=provider)

# Use HuggingFace
client = InferenceClient(token="your-hf-token")
response = client.text_generation("Hello, how are you?", model="meta-llama/Llama-2-7b-chat-hf")
print(response)
```

### Text Generation

```python
from huggingface_hub import InferenceClient

client = InferenceClient()

# Simple text generation
response = client.text_generation(
    "The capital of France is",
    model="meta-llama/Llama-2-7b-chat-hf",
    max_new_tokens=100,
    temperature=0.7,
)
print(response)

# With more parameters
response = client.text_generation(
    "Write a poem about Python programming",
    model="meta-llama/Llama-2-7b-chat-hf",
    max_new_tokens=200,
    temperature=0.9,
    top_p=0.95,
    repetition_penalty=1.1,
    do_sample=True,
)
print(response)
```

### Chat Completion

```python
from huggingface_hub import InferenceClient

client = InferenceClient()

# Chat completion with messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"}
]

response = client.chat_completion(
    messages=messages,
    model="meta-llama/Llama-2-7b-chat-hf",
    max_tokens=500,
    temperature=0.7,
)
print(response.choices[0].message.content)

# Multi-turn conversation
messages = [
    {"role": "user", "content": "My name is Alice"},
    {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
    {"role": "user", "content": "What's my name?"}
]

response = client.chat_completion(
    messages=messages,
    model="meta-llama/Llama-2-7b-chat-hf",
)
print(response.choices[0].message.content)
```

### Streaming

```python
from huggingface_hub import InferenceClient

client = InferenceClient()

# Streaming text generation
for chunk in client.text_generation(
    "Tell me a story about a brave knight",
    model="meta-llama/Llama-2-7b-chat-hf",
    max_new_tokens=500,
    stream=True,
):
    print(chunk, end="", flush=True)
print()

# Streaming chat completion
messages = [{"role": "user", "content": "Count to 10"}]
for chunk in client.chat_completion(
    messages=messages,
    model="meta-llama/Llama-2-7b-chat-hf",
    stream=True,
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

### Feature Extraction (Embeddings)

```python
from huggingface_hub import InferenceClient

client = InferenceClient()

# Single text embedding
embedding = client.feature_extraction(
    "Hello world",
    model="sentence-transformers/all-MiniLM-L6-v2",
)
print(f"Embedding dimensions: {len(embedding[0])}")

# Multiple texts
texts = ["Hello world", "Machine learning is great", "Python is awesome"]
embeddings = [
    client.feature_extraction(text, model="sentence-transformers/all-MiniLM-L6-v2")
    for text in texts
]
print(f"Generated {len(embeddings)} embeddings")
```

### Async Client

```python

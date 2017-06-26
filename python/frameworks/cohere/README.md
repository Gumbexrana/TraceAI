# TraceAI Cohere Instrumentation

OpenTelemetry instrumentation for [Cohere](https://cohere.com/) - embeddings, chat, and rerank APIs.

## Installation

```bash
pip install traceai-cohere
```

## Features

- Automatic tracing of Cohere API calls
- Support for chat, embed, and rerank endpoints
- Streaming response support
- Token usage tracking
- Rerank relevance scores
- Full OpenTelemetry semantic conventions compliance

## Usage

### Basic Setup

```python
import cohere
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from traceai_cohere import CohereInstrumentor

# Set up tracing
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# Instrument Cohere
CohereInstrumentor().instrument(tracer_provider=provider)

# Use Cohere
client = cohere.Client(api_key="your-api-key")
response = client.chat(message="Hello!")
print(response.text)
```

### Chat

```python
import cohere

client = cohere.Client()

# Simple chat
response = client.chat(
    model="command-r-plus",
    message="What is machine learning?"
)
print(response.text)

# With chat history
response = client.chat(
    model="command-r-plus",
    message="What's my name?",
    chat_history=[
        {"role": "USER", "message": "My name is Alice"},
        {"role": "CHATBOT", "message": "Hello Alice!"}
    ]
)

# With preamble (system prompt)
response = client.chat(
    model="command-r-plus",
    message="Write a poem",
    preamble="You are a creative poet who writes in haiku format."
)
```

### Streaming Chat

```python
import cohere

client = cohere.Client()

# Streaming
for event in client.chat_stream(
    model="command-r-plus",
    message="Tell me a story"
):
    if event.event_type == "text-generation":
        print(event.text, end="", flush=True)
```

### Embeddings

```python
import cohere

client = cohere.Client()

# Generate embeddings
response = client.embed(
    model="embed-english-v3.0",
    texts=["Hello world", "Machine learning is great"],
    input_type="search_document"
)
print(f"Generated {len(response.embeddings)} embeddings")
print(f"Dimensions: {len(response.embeddings[0])}")
```

### Rerank (for RAG)

```python
import cohere

client = cohere.Client()

# Rerank documents for a query
query = "What is the capital of France?"
documents = [
    "Paris is the capital of France.",
    "London is the capital of England.",
    "The Eiffel Tower is in Paris.",
    "France is a country in Europe."
]

response = client.rerank(
    model="rerank-english-v3.0",
    query=query,
    documents=documents,
    top_n=3
)

for result in response.results:
    print(f"Index: {result.index}, Score: {result.relevance_score:.4f}")
    print(f"  {documents[result.index]}")
```

### RAG with Cohere

```python
import cohere

client = cohere.Client()

# Documents for context
documents = [
    {"title": "Paris", "text": "Paris is the capital of France."},
    {"title": "London", "text": "London is the capital of England."},
]

response = client.chat(
    model="command-r-plus",
    message="What is the capital of France?",
    documents=documents
)

print(response.text)
if response.citations:
    print("\nCitations:")
    for citation in response.citations:
        print(f"  - {citation}")
```

### Tool Use

```python
import cohere

client = cohere.Client()

tools = [
    {
        "name": "get_weather",
        "description": "Get the weather for a location",
        "parameter_definitions": {
            "location": {
                "type": "str",
                "description": "The city name",
                "required": True
            }
        }
    }
]

response = client.chat(
    model="command-r-plus",
    message="What's the weather in Paris?",
    tools=tools
)

if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"Tool: {tool_call.name}")
        print(f"Parameters: {tool_call.parameters}")
```

## Configuration Options

### TraceConfig

```python
from fi_instrumentation import TraceConfig
from traceai_cohere import CohereInstrumentor

config = TraceConfig(
    hide_inputs=False,
    hide_outputs=False,
)

CohereInstrumentor().instrument(
    tracer_provider=provider,
    config=config
)
```

## Captured Attributes

### Chat Attributes

| Attribute | Description |

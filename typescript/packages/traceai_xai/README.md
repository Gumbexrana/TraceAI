# @traceai/xai

OpenTelemetry instrumentation for xAI (Grok) - AI models from xAI.

## Installation

```bash
npm install @traceai/xai
```

## Features

- Automatic tracing of xAI API calls through OpenAI-compatible interface
- Support for chat completions and embeddings
- Streaming response support
- Tool/function calling support
- Token usage tracking
- Full OpenTelemetry semantic conventions compliance

## Usage

### Basic Setup

```typescript
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { SimpleSpanProcessor, ConsoleSpanExporter } from "@opentelemetry/sdk-trace-base";
import { XAIInstrumentation } from "@traceai/xai";
import OpenAI from "openai";

// Set up the tracer provider
const provider = new NodeTracerProvider();
provider.addSpanProcessor(new SimpleSpanProcessor(new ConsoleSpanExporter()));
provider.register();

// Create and enable the instrumentation
const instrumentation = new XAIInstrumentation();
instrumentation.setTracerProvider(provider);
instrumentation.enable();

// Manually instrument the OpenAI module
const openaiModule = await import("openai");
instrumentation.manuallyInstrument(openaiModule);

// Create the xAI client
const client = new OpenAI({
  baseURL: "https://api.x.ai/v1",
  apiKey: process.env.XAI_API_KEY,
});

// Make requests - they will be automatically traced
const response = await client.chat.completions.create({
  model: "grok-beta",
  messages: [
    { role: "system", content: "You are Grok, a witty AI assistant." },
    { role: "user", content: "Hello!" },
  ],
});
```

### Streaming Responses

```typescript
const stream = await client.chat.completions.create({
  model: "grok-beta",
  messages: [{ role: "user", content: "Tell me a joke." }],
  stream: true,
});

for await (const chunk of stream) {

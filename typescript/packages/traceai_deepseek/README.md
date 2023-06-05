# @traceai/deepseek

OpenTelemetry instrumentation for DeepSeek - AI models with advanced reasoning capabilities.

## Installation

```bash
npm install @traceai/deepseek
```

## Features

- Automatic tracing of DeepSeek API calls through OpenAI-compatible interface
- Support for chat completions with streaming
- DeepSeek R1 reasoning content capture
- Prompt cache hit/miss metrics
- Tool/function calling support
- Full OpenTelemetry semantic conventions compliance

## Usage

### Basic Setup

```typescript
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { SimpleSpanProcessor, ConsoleSpanExporter } from "@opentelemetry/sdk-trace-base";
import { DeepSeekInstrumentation } from "@traceai/deepseek";
import OpenAI from "openai";

// Set up the tracer provider
const provider = new NodeTracerProvider();
provider.addSpanProcessor(new SimpleSpanProcessor(new ConsoleSpanExporter()));
provider.register();

// Create and enable the instrumentation
const instrumentation = new DeepSeekInstrumentation();
instrumentation.setTracerProvider(provider);
instrumentation.enable();

// Manually instrument the OpenAI module
const openaiModule = await import("openai");
instrumentation.manuallyInstrument(openaiModule);

// Create the DeepSeek client
const client = new OpenAI({
  baseURL: "https://api.deepseek.com/v1",
  apiKey: process.env.DEEPSEEK_API_KEY,
});

// Make requests - they will be automatically traced
const response = await client.chat.completions.create({
  model: "deepseek-chat",
  messages: [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "Hello!" },
  ],
});
```

### DeepSeek R1 Reasoning Models


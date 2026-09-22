# @traceai/vllm

OpenTelemetry instrumentation for vLLM - high-throughput and memory-efficient inference engine for LLMs.

## Installation

```bash
npm install @traceai/vllm
```

## Features

- Automatic tracing of vLLM API calls through OpenAI-compatible interface
- Support for chat completions and text completions
- Streaming response support
- Tool/function calling support
- Configurable URL pattern matching for multi-endpoint setups
- Token usage tracking
- Full OpenTelemetry semantic conventions compliance

## Usage

### Basic Setup

```typescript
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { SimpleSpanProcessor, ConsoleSpanExporter } from "@opentelemetry/sdk-trace-base";
import { VLLMInstrumentation } from "@traceai/vllm";
import OpenAI from "openai";

// Set up the tracer provider
const provider = new NodeTracerProvider();
provider.addSpanProcessor(new SimpleSpanProcessor(new ConsoleSpanExporter()));
provider.register();

// Create and enable the instrumentation
const instrumentation = new VLLMInstrumentation({
  // Optional: Only trace requests to specific vLLM servers
  baseUrlPattern: "localhost:8000",
});
instrumentation.setTracerProvider(provider);
instrumentation.enable();

// Manually instrument the OpenAI module
const openaiModule = await import("openai");
instrumentation.manuallyInstrument(openaiModule);

// Create the vLLM client
const client = new OpenAI({
  baseURL: "http://localhost:8000/v1",
  apiKey: "not-needed", // vLLM doesn't require API key
});

// Make requests - they will be automatically traced
const response = await client.chat.completions.create({
  model: "meta-llama/Llama-2-7b-chat-hf",
  messages: [
    { role: "system", content: "You are a helpful assistant." },

# @traceai/mistral

OpenTelemetry instrumentation for [Mistral AI](https://mistral.ai/) - leading open-weight AI models.

## Features

- Automatic tracing of Mistral Chat Completions
- Embeddings API tracing
- Support for all Mistral models
- Tool/function calling tracing
- Token usage tracking
- Error handling and exception recording

## Installation

```bash
npm install @traceai/mistral
# or
pnpm add @traceai/mistral
# or
yarn add @traceai/mistral
```

## Quick Start

```typescript
import { MistralInstrumentation } from "@traceai/mistral";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { Mistral } from "@mistralai/mistralai";

// Initialize OpenTelemetry
const provider = new NodeTracerProvider();
const exporter = new OTLPTraceExporter({
  url: "http://localhost:4318/v1/traces",
});
provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
provider.register();

// Initialize Mistral instrumentation
const mistralInstrumentation = new MistralInstrumentation();
mistralInstrumentation.manuallyInstrument(require("@mistralai/mistralai"));

// Use Mistral as normal - all calls are automatically traced
const mistral = new Mistral({ apiKey: process.env.MISTRAL_API_KEY });

const chatResponse = await mistral.chat.complete({
  model: "mistral-large-latest",
  messages: [
    { role: "user", content: "Explain quantum computing in simple terms" }
  ],
});

console.log(chatResponse.choices[0].message.content);
```

## Configuration

### Basic Configuration

```typescript
const instrumentation = new MistralInstrumentation({
  instrumentationConfig: {
    enabled: true,
  },
});
```

### With Trace Configuration

```typescript
const instrumentation = new MistralInstrumentation({
  instrumentationConfig: {
    enabled: true,
  },
  traceConfig: {
    hideInputs: false,  // Set to true to hide sensitive input data
    hideOutputs: false, // Set to true to hide sensitive output data
  },
});
```

### With TraceAI Core

```typescript
import { register, ProjectType } from "@traceai/fi-core";
import { MistralInstrumentation } from "@traceai/mistral";
import { registerInstrumentations } from "@opentelemetry/instrumentation";

// 1. Register TraceAI Core TracerProvider
const tracerProvider = register({
  projectName: "my-mistral-app",
  projectType: ProjectType.OBSERVE,
  sessionName: "mistral-session-" + Date.now(),
});

// 2. Register instrumentation BEFORE importing Mistral SDK
registerInstrumentations({
  tracerProvider: tracerProvider,
  instrumentations: [new MistralInstrumentation()],
});

// 3. NOW import and use Mistral
const { Mistral } = await import("@mistralai/mistralai");
const client = new Mistral({ apiKey: process.env.MISTRAL_API_KEY });

// 4. Use normally - all calls are traced
const response = await client.chat.complete({
  model: "mistral-small-latest",
  messages: [{ role: "user", content: "Hello!" }],
});

// 5. Shutdown when done
await tracerProvider.shutdown();
```

## Supported Models

### Chat Models

| Model | ID | Description |
|-------|-----|-------------|
| Mistral Large | `mistral-large-latest` | Most capable model, flagship |
| Mistral Medium | `mistral-medium-latest` | Balanced performance |
| Mistral Small | `mistral-small-latest` | Fast and cost-effective |
| Open Mistral 7B | `open-mistral-7b` | Open-weight 7B model |
| Open Mixtral 8x7B | `open-mixtral-8x7b` | Open-weight MoE model |
| Open Mixtral 8x22B | `open-mixtral-8x22b` | Largest open model |
| Codestral | `codestral-latest` | Specialized for code |

### Embedding Models

| Model | ID | Dimensions |
|-------|-----|------------|
| Mistral Embed | `mistral-embed` | 1024 |

## Real-World Use Cases

### 1. AI Assistant with Function Calling

```typescript
async function aiAssistant(userQuery: string) {
  const tools = [
    {
      type: "function",
      function: {
        name: "get_current_weather",
        description: "Get the current weather in a location",
        parameters: {
          type: "object",
          properties: {
            location: {
              type: "string",
              description: "City name",
            },

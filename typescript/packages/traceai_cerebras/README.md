# @traceai/cerebras

OpenTelemetry instrumentation for Cerebras - ultra-fast inference on Wafer-Scale Engine hardware.

## Installation

```bash
npm install @traceai/cerebras
```

## Features

- Automatic tracing of Cerebras Cloud SDK calls
- Streaming response support
- Cerebras-specific time metrics (queue time, prompt time, completion time)
- Token usage tracking
- Full OpenTelemetry semantic conventions compliance

## Usage

### Basic Setup

```typescript
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { SimpleSpanProcessor, ConsoleSpanExporter } from "@opentelemetry/sdk-trace-base";
import { CerebrasInstrumentation } from "@traceai/cerebras";
import Cerebras from "@cerebras/cerebras_cloud_sdk";

// Set up the tracer provider
const provider = new NodeTracerProvider();
provider.addSpanProcessor(new SimpleSpanProcessor(new ConsoleSpanExporter()));
provider.register();

// Create and enable the instrumentation
const instrumentation = new CerebrasInstrumentation();
instrumentation.setTracerProvider(provider);
instrumentation.enable();

// Manually instrument the Cerebras module
const cerebrasModule = await import("@cerebras/cerebras_cloud_sdk");
instrumentation.manuallyInstrument(cerebrasModule);

// Create the Cerebras client
const client = new Cerebras({
  apiKey: process.env.CEREBRAS_API_KEY,
});

// Make requests - they will be automatically traced
const response = await client.chat.completions.create({
  model: "llama3.1-8b",
  messages: [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "Hello!" },
  ],
});
```

### Performance Metrics

Cerebras provides detailed timing information for analyzing inference performance:


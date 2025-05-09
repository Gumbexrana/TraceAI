# @traceai/ollama

OpenTelemetry instrumentation for the [Ollama](https://ollama.ai/) TypeScript/JavaScript SDK.

Ollama enables running large language models locally, providing privacy-first AI capabilities with support for Llama, Mistral, Phi, and many other open-source models.

## Installation

```bash
npm install @traceai/ollama
# or
pnpm add @traceai/ollama
# or
yarn add @traceai/ollama
```

## Features

- Automatic instrumentation of Ollama API calls
- Supports chat, generate, and embed APIs
- Captures request parameters, model information, and responses
- Full streaming support with token tracking
- Configurable input/output hiding for privacy
- Compatible with OpenTelemetry ecosystem

## Usage

### Basic Setup

```typescript
import { OllamaInstrumentation } from "@traceai/ollama";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { registerInstrumentations } from "@opentelemetry/instrumentation";

// Set up the tracer provider
const provider = new NodeTracerProvider();
provider.register();

// Register the Ollama instrumentation
registerInstrumentations({
  instrumentations: [new OllamaInstrumentation()],
});

// Now use the Ollama SDK as normal
import { Ollama } from "ollama";

const ollama = new Ollama({ host: "http://localhost:11434" });

// Chat example
const chatResponse = await ollama.chat({
  model: "llama3.2",
  messages: [{ role: "user", content: "Hello, how are you?" }],
});

// Generate example
const generateResponse = await ollama.generate({
  model: "llama3.2",
  prompt: "The capital of France is",
});

// Embed example
const embedResponse = await ollama.embed({
  model: "nomic-embed-text",
  input: ["Hello", "World"],
});
```

### Manual Instrumentation

If automatic instrumentation doesn't work (e.g., with bundlers), you can manually instrument:

```typescript
import { OllamaInstrumentation } from "@traceai/ollama";
import * as OllamaModule from "ollama";

const instrumentation = new OllamaInstrumentation();
instrumentation.manuallyInstrument(OllamaModule);
```

### Configuration Options

```typescript
const instrumentation = new OllamaInstrumentation({
  // OpenTelemetry instrumentation config
  instrumentationConfig: {
    enabled: true,
  },
  // TraceAI-specific config
  traceConfig: {
    hideInputs: false, // Set to true to hide input content in traces
    hideOutputs: false, // Set to true to hide output content in traces
  },
});
```

### With TraceAI Core

```typescript
import { register, ProjectType } from "@traceai/fi-core";
import { OllamaInstrumentation } from "@traceai/ollama";
import { registerInstrumentations } from "@opentelemetry/instrumentation";

// 1. Register TraceAI Core TracerProvider
const tracerProvider = register({
  projectName: "my-ollama-app",
  projectType: ProjectType.OBSERVE,
  sessionName: "ollama-session-" + Date.now(),
});

// 2. Register instrumentation BEFORE importing Ollama SDK
registerInstrumentations({
  tracerProvider: tracerProvider,
  instrumentations: [new OllamaInstrumentation()],
});

// 3. NOW import and use Ollama
const { Ollama } = await import("ollama");
const client = new Ollama();

// 4. Use normally - all calls are traced
const response = await client.chat({
  model: "llama3.2",
  messages: [{ role: "user", content: "Hello!" }],
});

// 5. Shutdown when done
await tracerProvider.shutdown();
```

## Real-World Use Cases

### 1. Local Chatbot with History

```typescript
import { Ollama } from "ollama";

const ollama = new Ollama();

interface Message {
  role: "system" | "user" | "assistant";
  content: string;
}

async function chat(message: string, history: Message[]) {
  const response = await ollama.chat({
    model: "llama3.2",
    messages: [
      { role: "system", content: "You are a helpful coding assistant." },
      ...history,
      { role: "user", content: message },
    ],
  });

  // Add to history
  history.push({ role: "user", content: message });
  history.push({ role: "assistant", content: response.message.content });

  return response.message.content;
}

// Usage
const history: Message[] = [];
const answer1 = await chat("How do I read a file in Python?", history);
const answer2 = await chat("What about async file reading?", history);
```

### 2. Streaming Chat Responses

```typescript
async function streamChat(message: string) {
  const stream = await ollama.chat({
    model: "llama3.2",
    messages: [{ role: "user", content: message }],
    stream: true,
  });

  let fullText = "";
  for await (const chunk of stream) {
    process.stdout.write(chunk.message.content);
    fullText += chunk.message.content;
  }
  console.log("\n--- Stream complete ---");

  return fullText;
}
```

### 3. Local Embeddings for Semantic Search

```typescript
async function semanticSearch(query: string, documents: string[]) {
  // Get query embedding
  const queryResult = await ollama.embed({
    model: "nomic-embed-text",
    input: query,
  });
  const queryVector = queryResult.embeddings[0];

  // Get document embeddings
  const docResult = await ollama.embed({
    model: "nomic-embed-text",
    input: documents,
  });

  // Calculate cosine similarities
  const similarities = docResult.embeddings.map((docVector, i) => ({
    document: documents[i],
    score: cosineSimilarity(queryVector, docVector),
  }));

  return similarities.sort((a, b) => b.score - a.score);
}

function cosineSimilarity(a: number[], b: number[]): number {
  const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}
```

### 4. Text Completion with System Prompt

```typescript
async function generateWithContext(prompt: string, systemPrompt: string) {
  const response = await ollama.generate({
    model: "llama3.2",
    prompt: prompt,
    system: systemPrompt,
    options: {

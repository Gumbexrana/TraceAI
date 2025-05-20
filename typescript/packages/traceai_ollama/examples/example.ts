/**
 * @traceai/ollama Real-World Example
 *
 * This example demonstrates how to use the Ollama instrumentation with TraceAI
 * for observability of local LLM calls including chat, generate, and embeddings.
 *
 * Prerequisites:
 * 1. Install Ollama: https://ollama.ai
 * 2. Start Ollama: ollama serve
 * 3. Pull models:
 *    - ollama pull llama3.2
 *    - ollama pull nomic-embed-text
 * 4. Set environment variables (see .env.example)
 * 5. Install dependencies: pnpm install
 * 6. Run: npx ts-node example.ts
 */

import { register, ProjectType } from "@traceai/fi-core";
import { OllamaInstrumentation } from "@traceai/ollama";
import { registerInstrumentations } from "@opentelemetry/instrumentation";
import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
import "dotenv/config";

// Enable OpenTelemetry diagnostics for debugging
diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.INFO);

// Validate environment variables
const fiApiKey = process.env.FI_API_KEY;
const fiSecretKey = process.env.FI_SECRET_KEY;
const ollamaHost = process.env.OLLAMA_HOST || "http://localhost:11434";
const ollamaModel = process.env.OLLAMA_MODEL || "llama3.2";
const ollamaEmbedModel = process.env.OLLAMA_EMBED_MODEL || "nomic-embed-text";

if (!fiApiKey || !fiSecretKey) {
  console.error("FI_API_KEY and FI_SECRET_KEY environment variables must be set.");
  process.exit(1);
}

async function main() {
  console.log("Starting Ollama Instrumentation Example...\n");
  console.log(`Ollama Host: ${ollamaHost}`);
  console.log(`Chat Model: ${ollamaModel}`);
  console.log(`Embed Model: ${ollamaEmbedModel}\n`);

  // 1. Register TraceAI Core TracerProvider
  const tracerProvider = register({
    projectName: "ollama-instrumentation-example",
    projectType: ProjectType.OBSERVE,
    sessionName: "ollama-example-session-" + Date.now(),
  });
  console.log("TraceAI Core TracerProvider registered.");

  // 2. Register Ollama Instrumentation BEFORE importing the SDK
  registerInstrumentations({
    tracerProvider: tracerProvider,
    instrumentations: [
      new OllamaInstrumentation({
        traceConfig: {
          hideInputs: false,
          hideOutputs: false,
        },
      }),
    ],
  });
  console.log("Ollama Instrumentation registered.\n");

  // 3. Dynamically import Ollama SDK AFTER instrumentation is registered
  const { Ollama } = await import("ollama");
  const client = new Ollama({ host: ollamaHost });

  try {
    // === Test Case 1: Basic Chat ===
    console.log("--- Test Case 1: Basic Chat ---");
    const basicChatResponse = await client.chat({
      model: ollamaModel,
      messages: [
        { role: "user", content: "What is the capital of France? Answer in one sentence." }
      ],
      stream: false,
    });
    console.log("Response:", basicChatResponse.message.content);
    console.log("Tokens - Prompt:", basicChatResponse.prompt_eval_count, "Completion:", basicChatResponse.eval_count);
    console.log();

    // === Test Case 2: Chat with System Prompt ===
    console.log("--- Test Case 2: Chat with System Prompt ---");
    const systemPromptResponse = await client.chat({
      model: ollamaModel,
      messages: [
        { role: "system", content: "You are a helpful assistant that speaks like a pirate." },
        { role: "user", content: "Tell me about the ocean." }
      ],
      stream: false,
    });
    console.log("Response:", systemPromptResponse.message.content);
    console.log();

    // === Test Case 3: Multi-turn Chat with History ===
    console.log("--- Test Case 3: Multi-turn Chat with History ---");

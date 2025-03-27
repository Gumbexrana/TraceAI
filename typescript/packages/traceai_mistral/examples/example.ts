/**
 * @traceai/mistral Real-World Example
 *
 * This example demonstrates how to use the Mistral instrumentation with TraceAI
 * for observability of Mistral API calls.
 *
 * Prerequisites:
 * 1. Set environment variables (see .env.example)
 * 2. Install dependencies: pnpm install
 * 3. Run: npx ts-node example.ts
 */

import { register, ProjectType } from "@traceai/fi-core";
import { MistralInstrumentation } from "@traceai/mistral";
import { registerInstrumentations } from "@opentelemetry/instrumentation";
import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
import "dotenv/config";

// Enable OpenTelemetry diagnostics for debugging
diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.INFO);

// Validate environment variables
const fiApiKey = process.env.FI_API_KEY;
const fiSecretKey = process.env.FI_SECRET_KEY;
const mistralApiKey = process.env.MISTRAL_API_KEY;

if (!fiApiKey || !fiSecretKey) {
  console.error("FI_API_KEY and FI_SECRET_KEY environment variables must be set.");
  process.exit(1);
}

if (!mistralApiKey) {
  console.error("MISTRAL_API_KEY environment variable must be set.");
  process.exit(1);
}

async function main() {
  console.log("Starting Mistral Instrumentation Example...\n");

  // 1. Register TraceAI Core TracerProvider
  const tracerProvider = register({
    projectName: "mistral-instrumentation-example",
    projectType: ProjectType.OBSERVE,
    sessionName: "mistral-example-session-" + Date.now(),
  });
  console.log("TraceAI Core TracerProvider registered.");

  // 2. Register Mistral Instrumentation BEFORE importing the SDK
  registerInstrumentations({
    tracerProvider: tracerProvider,
    instrumentations: [
      new MistralInstrumentation({
        traceConfig: {
          hideInputs: false,
          hideOutputs: false,
        },
      }),
    ],
  });
  console.log("Mistral Instrumentation registered.\n");

  // 3. Dynamically import Mistral SDK AFTER instrumentation is registered
  const { Mistral } = await import("@mistralai/mistralai");
  const client = new Mistral({ apiKey: mistralApiKey });

  try {
    // === Test Case 1: Basic Chat Completion ===
    console.log("--- Test Case 1: Basic Chat Completion ---");
    const basicResponse = await client.chat.complete({
      model: "mistral-small-latest",
      messages: [
        { role: "user", content: "What is the capital of Japan? Answer in one sentence." }
      ],
    });
    console.log("Response:", basicResponse.choices?.[0]?.message?.content);
    console.log("Usage:", basicResponse.usage);
    console.log();

    // === Test Case 2: Chat with System Prompt ===
    console.log("--- Test Case 2: Chat with System Prompt ---");

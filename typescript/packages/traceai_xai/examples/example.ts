/**
 * Example: Using xAI (Grok) instrumentation with TraceAI
 *
 * This example demonstrates how to set up and use the @traceai/xai
 * instrumentation to trace xAI API calls.
 *
 * xAI uses OpenAI-compatible API, so you use the OpenAI SDK with a custom baseURL.
 */
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { SimpleSpanProcessor, ConsoleSpanExporter } from "@opentelemetry/sdk-trace-base";
import { XAIInstrumentation } from "@traceai/xai";
import OpenAI from "openai";

async function main() {
  // Step 1: Set up the tracer provider
  const provider = new NodeTracerProvider();
  provider.addSpanProcessor(new SimpleSpanProcessor(new ConsoleSpanExporter()));
  provider.register();

  // Step 2: Create and enable the instrumentation
  const instrumentation = new XAIInstrumentation({
    traceConfig: {
      // Optional: Configure tracing options
      // hideInputs: true,  // Hide input content in traces
      // hideOutputs: true, // Hide output content in traces
    },
  });
  instrumentation.setTracerProvider(provider);
  instrumentation.enable();

  // Step 3: Import and patch the OpenAI module
  const openaiModule = await import("openai");
  instrumentation.manuallyInstrument(openaiModule);

  // Step 4: Create the xAI client (OpenAI SDK with custom baseURL)
  const client = new OpenAI({
    baseURL: "https://api.x.ai/v1",
    apiKey: process.env.XAI_API_KEY,
  });

  console.log("=== Chat Completion Example ===\n");

  // Example 1: Chat completion with Grok
  const chatResponse = await client.chat.completions.create({
    model: "grok-beta",
    messages: [
      { role: "system", content: "You are Grok, a witty AI assistant." },
      { role: "user", content: "What is the capital of France?" },
    ],
    max_tokens: 100,
    temperature: 0.7,
  });

  console.log("Chat Response:", chatResponse.choices[0]?.message?.content);

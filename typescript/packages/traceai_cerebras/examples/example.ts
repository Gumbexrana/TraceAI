/**
 * Example: Using Cerebras instrumentation with TraceAI
 *
 * This example demonstrates how to set up and use the @traceai/cerebras
 * instrumentation to trace Cerebras Cloud SDK calls.
 *
 * Cerebras uses its own SDK (@cerebras/cerebras_cloud_sdk), not the OpenAI SDK.
 */
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { SimpleSpanProcessor, ConsoleSpanExporter } from "@opentelemetry/sdk-trace-base";
import { CerebrasInstrumentation } from "@traceai/cerebras";
import Cerebras from "@cerebras/cerebras_cloud_sdk";

async function main() {
  // Step 1: Set up the tracer provider
  const provider = new NodeTracerProvider();
  provider.addSpanProcessor(new SimpleSpanProcessor(new ConsoleSpanExporter()));
  provider.register();

  // Step 2: Create and enable the instrumentation
  const instrumentation = new CerebrasInstrumentation({
    traceConfig: {
      // Optional: Configure tracing options
      // hideInputs: true,  // Hide input content in traces
      // hideOutputs: true, // Hide output content in traces
    },
  });
  instrumentation.setTracerProvider(provider);
  instrumentation.enable();

  // Step 3: Import and patch the Cerebras module
  const cerebrasModule = await import("@cerebras/cerebras_cloud_sdk");
  instrumentation.manuallyInstrument(cerebrasModule);

  // Step 4: Create the Cerebras client
  const client = new Cerebras({
    apiKey: process.env.CEREBRAS_API_KEY,
  });

  console.log("=== Chat Completion Example ===\n");

  // Example 1: Chat completion
  const chatResponse = await client.chat.completions.create({
    model: "llama3.1-8b",
    messages: [
      { role: "system", content: "You are a helpful assistant." },

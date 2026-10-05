/**
 * Example: Using vLLM instrumentation with TraceAI
 *
 * This example demonstrates how to set up and use the @traceai/vllm
 * instrumentation to trace vLLM inference calls.
 *
 * vLLM uses OpenAI-compatible API, so you use the OpenAI SDK with a custom baseURL.
 */
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { SimpleSpanProcessor, ConsoleSpanExporter } from "@opentelemetry/sdk-trace-base";
import { VLLMInstrumentation } from "@traceai/vllm";
import OpenAI from "openai";

async function main() {
  // Step 1: Set up the tracer provider
  const provider = new NodeTracerProvider();
  provider.addSpanProcessor(new SimpleSpanProcessor(new ConsoleSpanExporter()));
  provider.register();

  // Step 2: Create and enable the instrumentation
  const instrumentation = new VLLMInstrumentation({
    // Optional: Configure a URL pattern to identify vLLM requests
    // If not set, all OpenAI SDK requests will be traced as vLLM
    baseUrlPattern: "localhost:8000",
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

  // Step 4: Create the vLLM client (OpenAI SDK with custom baseURL)
  const client = new OpenAI({

/**
 * Example of using Instructor instrumentation with FI tracing.
 *
 * This example shows how to instrument the Instructor library
 * for observability with OpenTelemetry.
 */

import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { SimpleSpanProcessor, ConsoleSpanExporter } from "@opentelemetry/sdk-trace-base";
import { InstructorInstrumentation } from "@traceai/fi-instrumentation-instructor";

// Set up OpenTelemetry tracing
const provider = new NodeTracerProvider();
provider.addSpanProcessor(new SimpleSpanProcessor(new ConsoleSpanExporter()));
provider.register();

// Create and register the instrumentation
const instrumentation = new InstructorInstrumentation({
  traceConfig: {
    hideInputs: false,
    hideOutputs: false,
  },
});
instrumentation.setTracerProvider(provider);
instrumentation.enable();

async function main() {
  // Import Instructor after instrumentation is set up
  // const Instructor = (await import("@instructor-ai/instructor")).default;
  // const OpenAI = (await import("openai")).default;

  console.log("Instructor instrumentation example");
  console.log("==================================");

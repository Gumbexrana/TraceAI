/**
 * Example of using Portkey instrumentation with FI tracing.
 *
 * This example shows how to instrument the Portkey AI Gateway SDK
 * for observability with OpenTelemetry.
 */

import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { SimpleSpanProcessor, ConsoleSpanExporter } from "@opentelemetry/sdk-trace-base";
import { PortkeyInstrumentation } from "@traceai/fi-instrumentation-portkey";

// Set up OpenTelemetry tracing
const provider = new NodeTracerProvider();
provider.addSpanProcessor(new SimpleSpanProcessor(new ConsoleSpanExporter()));
provider.register();

// Create and register the instrumentation
const instrumentation = new PortkeyInstrumentation({
  traceConfig: {
    hideInputs: false,
    hideOutputs: false,
  },
});
instrumentation.setTracerProvider(provider);
instrumentation.enable();

async function main() {
  // Import Portkey after instrumentation is set up
  // const { Portkey } = await import("portkey-ai");

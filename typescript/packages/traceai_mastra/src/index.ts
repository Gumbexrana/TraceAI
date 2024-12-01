// Mastra v1 (>= 1.16) — recommended. A custom observability exporter that maps
// Mastra spans to Future AGI's gen_ai conventions (span kind + input/output) and
// ships OTLP to the collector. Depends only on the @mastra/* peers — no legacy
// OpenTelemetry/Vercel deps are loaded.
export {
  createFIObservability,
  createFIMastraExporter,

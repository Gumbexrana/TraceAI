import { BaseExporter } from "@mastra/observability";
import { SpanConverter } from "@mastra/otel-exporter";
import { SpanType } from "@mastra/core/observability";
import type { AnyExportedSpan, TracingEvent } from "@mastra/core/observability";
import { FISpanKind } from "@traceai/fi-semantic-conventions";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { BatchSpanProcessor } from "@opentelemetry/sdk-trace-base";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";

export interface FISpanExporterConfig {
  /** Full OTLP traces endpoint URL. */
  endpoint: string;
  /** Auth + extra headers for the export request. */
  headers: Record<string, string>;
  /** Service name (also the resource `service.name`). */
  serviceName: string;
  /** OTel resource attributes (must include `project_name` / `project_type`). */
  resourceAttributes: Record<string, string>;
  /** Export request timeout in ms. */
  timeout?: number;
  /** Spans per export batch. */
  batchSize?: number;
}

/**
 * Mastra span type → Future AGI span kind (`gen_ai.span.kind`).
 *
 * The FI backend maps these (case-insensitively) to its observation types
 * (llm / agent / tool / chain / ...). Mastra emits `gen_ai.operation.name` but
 * NOT a span kind, so without this mapping every span renders as "unknown".
 */
const SPAN_KIND_BY_TYPE: Partial<Record<SpanType, FISpanKind>> = {
  [SpanType.AGENT_RUN]: FISpanKind.AGENT,
  [SpanType.MODEL_GENERATION]: FISpanKind.LLM,
  [SpanType.MODEL_INFERENCE]: FISpanKind.LLM,
  [SpanType.MODEL_STEP]: FISpanKind.CHAIN,
  [SpanType.TOOL_CALL]: FISpanKind.TOOL,
  [SpanType.MCP_TOOL_CALL]: FISpanKind.TOOL,
  [SpanType.CLIENT_TOOL_CALL]: FISpanKind.TOOL,
  [SpanType.WORKFLOW_RUN]: FISpanKind.CHAIN,
  [SpanType.WORKFLOW_STEP]: FISpanKind.CHAIN,
  [SpanType.WORKFLOW_CONDITIONAL]: FISpanKind.CHAIN,
  [SpanType.WORKFLOW_PARALLEL]: FISpanKind.CHAIN,
  [SpanType.WORKFLOW_LOOP]: FISpanKind.CHAIN,
  [SpanType.GENERIC]: FISpanKind.CHAIN,
  [SpanType.RAG_EMBEDDING]: FISpanKind.EMBEDDING,
  [SpanType.RAG_VECTOR_OPERATION]: FISpanKind.RETRIEVER,
};

/**
 * Write `<key>.value` + `<key>.mime_type` the way the FI backend's `set_io_value`
 * expects, so the trace UI renders the input/output.
 */
function setIoValue(
  attrs: Record<string, unknown>,
  key: "input" | "output",
  value: unknown,
): void {
  if (value === undefined || value === null) return;

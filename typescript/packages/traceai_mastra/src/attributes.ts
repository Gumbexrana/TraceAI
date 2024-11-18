import {
  FISpanKind,
  SemanticConventions,
} from "@traceai/fi-semantic-conventions";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";

const MASTRA_AGENT_SPAN_NAME_PREFIXES = [
  "agent",
  "mastra.getAgent",
  "post /api/agents",
];

/**
 * Add the FI span kind to the given Mastra span.
 *
 * This function will add the FI span kind to the given Mastra span.
 */
const addFISpanKind = (
  span: ReadableSpan,
  kind: FISpanKind,
) => {
  span.attributes[SemanticConventions.FI_SPAN_KIND] = kind;
};

/**
 * Get the FI span kind for the given Mastra span.
 *
 * This function will return the FI span kind for the given Mastra span, if it has already been set.
 */
const getFISpanKind = (span: ReadableSpan) => {

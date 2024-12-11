import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  SemanticConventions,
  ATTR_PROJECT_NAME,
} from "@traceai/fi-semantic-conventions";
import { ATTR_SERVICE_NAME } from "@opentelemetry/semantic-conventions";

/**
 * Augments a span with OpenInference project resource attribute.
 *
 * This function will add additional attributes to the span, based on the span's resource attributes.
 *
 * @param span - The span to augment.
 */

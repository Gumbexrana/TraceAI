import { Observability } from "@mastra/observability";
import { SpanType } from "@mastra/core/observability";
import {
  FIMastraSpanExporter,
  type FISpanExporterConfig,
} from "./FIMastraSpanExporter.js";


const DEFAULT_FI_BASE_URL = "https://api.futureagi.com";
const FI_TRACES_PATH = "/tracer/v1/traces";

export interface FIMastraExporterOptions {
  /** Service name (resource `service.name`). Defaults to `"mastra-app"`. */
  serviceName?: string;
  /** Future AGI API key. Defaults to `process.env.FI_API_KEY`. */
  apiKey?: string;
  /** Future AGI secret key. Defaults to `process.env.FI_SECRET_KEY`. */
  secretKey?: string;
  /**
   * Full traces endpoint URL. When set, overrides `baseUrl`.
   * Defaults to `${baseUrl}/tracer/v1/traces`.
   */
  endpoint?: string;
  /**
   * Base collector URL. The `/tracer/v1/traces` path is appended.
   * Defaults to `process.env.FI_BASE_URL` or `https://api.futureagi.com`.
   */
  baseUrl?: string;
  /** Extra headers merged into the export request. */
  headers?: Record<string, string>;
  /**
   * Future AGI project name. This is the `project_name` resource attribute the
   * Future AGI collector keys on — WITHOUT it, traces are ingested but no project
   * is created and nothing shows in the dashboard. Defaults to `FI_PROJECT_NAME`
   * env, then (via {@link createFIObservability}) the `serviceName`.
   */
  projectName?: string;
  /**
   * Future AGI project type. `"observe"` for continuous tracing (default),
   * `"experiment"` for evaluation runs with project versions.
   */
  projectType?: "observe" | "experiment";
  /** Extra OTel resource attributes merged onto every span's resource. */
  resourceAttributes?: Record<string, string>;
  /** Export request timeout in ms. */
  timeout?: number;
  /** Spans per export batch. */
  batchSize?: number;
}

/** @internal Exported for unit testing; not part of the public package API. */
export function resolveResourceAttributes(
  options: FIMastraExporterOptions,
): Record<string, string> {
  const projectName = options.projectName ?? process.env.FI_PROJECT_NAME;
  return {
    // Future AGI keys the project on these resource attributes.
    ...(projectName ? { project_name: projectName } : {}),
    project_type: options.projectType ?? "observe",
    ...(options.resourceAttributes ?? {}),
  };
}


/**
 * OpenTelemetry instrumentation for Instructor.
 *
 * Instructor is a library for structured output from LLMs using Pydantic models.
 * This instrumentation captures structured extraction calls.
 */

import {
  InstrumentationBase,
  InstrumentationNodeModuleDefinition,
} from "@opentelemetry/instrumentation";
import {
  Span,
  SpanKind,
  SpanStatusCode,
  context,
  trace,
} from "@opentelemetry/api";
import { FITracer, TraceConfigOptions } from "@traceai/fi-core";
import {
  SemanticConventions,
  FISpanKind,
  MimeType,
} from "@traceai/fi-semantic-conventions";
import { VERSION } from "./version";

export interface InstructorInstrumentationConfig {
  instrumentationConfig?: Record<string, unknown>;
  traceConfig?: TraceConfigOptions;
}

/**
 * Instructor Instrumentation class.
 *
 * Provides automatic instrumentation for the @instructor-ai/instructor SDK.
 */
export class InstructorInstrumentation extends InstrumentationBase {
  private fiTracer!: FITracer;
  private _traceConfig?: TraceConfigOptions;

  constructor(config: InstructorInstrumentationConfig = {}) {
    super(
      "@traceai/fi-instrumentation-instructor",
      VERSION,
      config.instrumentationConfig || {}
    );
    this._traceConfig = config.traceConfig;
  }

  override enable(): void {
    super.enable();
    this.fiTracer = new FITracer({
      tracer: this.tracer,
      traceConfig: this._traceConfig,
    });
  }

  protected init() {
    const module = new InstrumentationNodeModuleDefinition(
      "@instructor-ai/instructor",
      ["^1.0.0"],
      this.patch.bind(this),
      this.unpatch.bind(this)
    );
    return module;
  }

  /**
   * Manually instrument the @instructor-ai/instructor module.
   */
  manuallyInstrument(instructorModule: any): void {
    if (instructorModule == null) return;
    this.patch(instructorModule);
  }

  private patch(instructorModule: any & { _fiPatched?: boolean }): any {
    if (instructorModule?._fiPatched || _isFIPatched) {
      return instructorModule;
    }


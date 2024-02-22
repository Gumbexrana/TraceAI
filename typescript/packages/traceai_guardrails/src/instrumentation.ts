/**
 * OpenTelemetry instrumentation for Guardrails AI.
 *
 * Guardrails AI provides validation and structural guarantees for LLM outputs.
 * This instrumentation captures guard execution and validation results.
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

export interface GuardrailsInstrumentationConfig {
  instrumentationConfig?: Record<string, unknown>;
  traceConfig?: TraceConfigOptions;
}

/**
 * Guardrails AI Instrumentation class.
 *
 * Provides automatic instrumentation for the @guardrails-ai/core SDK.
 */
export class GuardrailsInstrumentation extends InstrumentationBase {
  private fiTracer!: FITracer;
  private _traceConfig?: TraceConfigOptions;

  constructor(config: GuardrailsInstrumentationConfig = {}) {
    super(
      "@traceai/fi-instrumentation-guardrails",
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
      "@guardrails-ai/core",
      ["^0.1.0"],
      this.patch.bind(this),
      this.unpatch.bind(this)
    );
    return module;
  }

  /**
   * Manually instrument the @guardrails-ai/core module.
   */
  manuallyInstrument(guardrailsModule: any): void {
    if (guardrailsModule == null) return;
    this.patch(guardrailsModule);
  }

  private patch(guardrailsModule: any & { _fiPatched?: boolean }): any {
    if (guardrailsModule?._fiPatched || _isFIPatched) {
      return guardrailsModule;
    }

    const instrumentation = this;

    // Wrap Guard.validate method
    if (guardrailsModule.Guard?.prototype?.validate) {
      this._wrap(
        guardrailsModule.Guard.prototype,
        "validate",
        (original: Function) => {
          return function patchedValidate(this: any, ...args: any[]) {
            return instrumentation.traceGuardValidation(original, this, args, "validate");

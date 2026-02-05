/**
 * OpenTelemetry instrumentation for Portkey AI Gateway.
 *
 * Portkey is an AI gateway that provides unified access to multiple LLM providers.
 * This instrumentation captures gateway requests and responses.
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
import { FITracer, TraceConfigOptions, safelyJSONStringify } from "@traceai/fi-core";
import {
  SemanticConventions,
  FISpanKind,
  MimeType,
} from "@traceai/fi-semantic-conventions";
import { VERSION } from "./version";

export interface PortkeyInstrumentationConfig {
  instrumentationConfig?: Record<string, unknown>;
  traceConfig?: TraceConfigOptions;
}

/**
 * Portkey Instrumentation class.
 *
 * Provides automatic instrumentation for the portkey-ai SDK.
 */
export class PortkeyInstrumentation extends InstrumentationBase {
  private fiTracer!: FITracer;
  private _traceConfig?: TraceConfigOptions;

  constructor(config: PortkeyInstrumentationConfig = {}) {
    super(
      "@traceai/fi-instrumentation-portkey",
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
      "portkey-ai",
      ["^1.0.0"],
      this.patch.bind(this),
      this.unpatch.bind(this)
    );
    return module;
  }

  /**
   * Manually instrument the portkey-ai module.
   */
  manuallyInstrument(portkeyModule: any): void {
    if (portkeyModule == null) return;
    this.patch(portkeyModule);
  }

  private patch(portkeyModule: any & { _fiPatched?: boolean }): any {
    if (portkeyModule?._fiPatched || _isFIPatched) {
      return portkeyModule;
    }

    const instrumentation = this;

    // Wrap Portkey chat completions
    if (portkeyModule.Portkey?.prototype?.chat?.completions?.create) {
      this._wrap(
        portkeyModule.Portkey.prototype.chat.completions,
        "create",
        (original: Function) => {
          return function patchedCreate(this: any, ...args: any[]) {
            return instrumentation.traceChatCompletion(original, this, args);
          };
        }
      );
    }

    // Wrap Portkey completions
    if (portkeyModule.Portkey?.prototype?.completions?.create) {
      this._wrap(
        portkeyModule.Portkey.prototype.completions,
        "create",
        (original: Function) => {
          return function patchedCreate(this: any, ...args: any[]) {
            return instrumentation.traceCompletion(original, this, args);
          };
        }
      );
    }

    // Wrap Portkey embeddings
    if (portkeyModule.Portkey?.prototype?.embeddings?.create) {
      this._wrap(
        portkeyModule.Portkey.prototype.embeddings,
        "create",
        (original: Function) => {
          return function patchedCreate(this: any, ...args: any[]) {
            return instrumentation.traceEmbedding(original, this, args);
          };
        }
      );
    }

    portkeyModule._fiPatched = true;
    _isFIPatched = true;
    return portkeyModule;
  }

  private unpatch(portkeyModule: any & { _fiPatched?: boolean }): void {
    if (portkeyModule?.Portkey?.prototype?.chat?.completions?.create) {
      this._unwrap(portkeyModule.Portkey.prototype.chat.completions, "create");
    }
    if (portkeyModule?.Portkey?.prototype?.completions?.create) {
      this._unwrap(portkeyModule.Portkey.prototype.completions, "create");
    }
    if (portkeyModule?.Portkey?.prototype?.embeddings?.create) {
      this._unwrap(portkeyModule.Portkey.prototype.embeddings, "create");
    }
    if (portkeyModule) {
      portkeyModule._fiPatched = false;
    }
    _isFIPatched = false;
  }

  private async traceChatCompletion(
    original: Function,
    instance: any,
    args: any[]
  ): Promise<any> {
    const params = args[0] || {};
    const model = params.model || "unknown";
    const messages = params.messages || [];

    // Build input messages as JSON blob
    const serializedMessages = messages.map((msg: any) => {
      const obj: Record<string, unknown> = {};
      if (msg.role) obj.role = msg.role;
      if (msg.content) obj.content = typeof msg.content === "string" ? msg.content : safeJsonStringify(msg.content);
      return obj;
    });

    const span = this.fiTracer.startSpan("Portkey Chat Completions", {
      kind: SpanKind.INTERNAL,
      attributes: {
        [SemanticConventions.FI_SPAN_KIND]: FISpanKind.LLM,
        [SemanticConventions.LLM_PROVIDER]: "portkey",

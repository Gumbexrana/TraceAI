import Anthropic from "@anthropic-ai/sdk";
import {
  InstrumentationBase,
  InstrumentationConfig,
  InstrumentationModuleDefinition,
  InstrumentationNodeModuleDefinition,
  safeExecuteInTheMiddle,
} from "@opentelemetry/instrumentation";
import {
  diag,
  context,
  trace,
  SpanKind,
  Attributes,
  SpanStatusCode,
  Span,
} from "@opentelemetry/api";
import { SemanticConventions, FISpanKind, LLMProvider, MimeType } from "@traceai/fi-semantic-conventions";
import { isTracingSuppressed } from "@opentelemetry/core";
import { FITracer, safelyJSONStringify, TraceConfigOptions } from "@traceai/fi-core";
import { VERSION } from "./version";
import {
    getAnthropicInputMessagesAttributes,
    getAnthropicOutputMessagesAttributes,
    getAnthropicUsageAttributes,
    getAnthropicToolsAttributes,
    aggregateAnthropicStreamEvents,
} from "./responseAttributes";
import { type MessageStreamEvent } from "@anthropic-ai/sdk/resources/messages";
import { Stream } from "@anthropic-ai/sdk/streaming";

const MODULE_NAME = "@anthropic-ai/sdk";
let _isFIPatched = false;

export function isPatched() {
  return _isFIPatched;
}

function getExecContext(span: Span) {
  const activeContext = context.active();
  const suppressTracing = isTracingSuppressed(activeContext);
  // If tracing is suppressed, don't set the span in context, effectively dropping it for nested operations
  const execContext = suppressTracing ? activeContext : trace.setSpan(context.active(), span);
  // TODO: check if this diag log is needed or should be removed/conditional
  // if (suppressTracing) {
  //   diag.debug("Tracing is suppressed. Span will not be exported or set in context for children.");
  // }
  return execContext;
}

export class AnthropicInstrumentation extends InstrumentationBase{
  private fiTracer!: FITracer;
  private _traceConfig?: TraceConfigOptions;

  constructor({
    instrumentationConfig,
    traceConfig,
  }: {
    instrumentationConfig?: InstrumentationConfig;
    traceConfig?: TraceConfigOptions;
  } = {}) {
    super("@traceai/instrumentation-anthropic", VERSION,Object.assign({}, instrumentationConfig));
    this._traceConfig = traceConfig;
  }

  public override enable(): void {
    super.enable();
    this.fiTracer = new FITracer({ tracer: this.tracer, traceConfig: this._traceConfig });
  }

  protected init(){
    const module = new InstrumentationNodeModuleDefinition(
      MODULE_NAME,
      [">=0.20.0 <1"], // Looser version range based on recent activity
      this.patch.bind(this),
      this.unpatch.bind(this),
    );
    return module;
  }

  manuallyInstrument(module: typeof Anthropic) {
    this.patch(module);
  }

  private patch(moduleExports: typeof Anthropic, moduleVersion?: string) {
    if ((moduleExports as any).fiPatched || _isFIPatched) {
      return moduleExports;
    }
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const instrumentation: AnthropicInstrumentation = this;

    if (!(instrumentation.fiTracer as any).tracer || (instrumentation.fiTracer as any).tracer !== instrumentation.tracer) {
        instrumentation.fiTracer = new FITracer({ tracer: instrumentation.tracer, traceConfig: instrumentation._traceConfig });
    }

    const messagesPrototype = moduleExports.Messages?.prototype;
    if (!messagesPrototype) {
      diag.error("Anthropic.Messages.prototype not found, cannot patch messages.create.");
      return moduleExports;
    }

    // Patch messages.create
    this._wrap(
      messagesPrototype,
      "create",

import type * as CallbackManagerModuleV02 from "@langchain/core/callbacks/manager";
import {
  InstrumentationBase,
  InstrumentationConfig,
  InstrumentationModuleDefinition,
  InstrumentationNodeModuleDefinition,
  isWrapped,
} from "@opentelemetry/instrumentation";
import { VERSION } from "./version";
import { diag, Tracer, TracerProvider } from "@opentelemetry/api";
import { addTracerToHandlers } from "./instrumentationUtils";
import { FITracer, TraceConfigOptions } from "@traceai/fi-core";

const MODULE_NAME = "@langchain/core/callbacks";

const INSTRUMENTATION_NAME = "@traceai/langchain";
/**
 * Flag to check if the openai module has been patched
 * Note: This is a fallback in case the module is made immutable (e.x. Deno, webpack, etc.)
 */
let _isFIPatched = false;

/**
 * function to check if instrumentation is enabled / disabled
 */
export function isPatched() {
  return _isFIPatched;
}

/**
 * Resets the internal patched state. Intended for testing only.
 */
export function _resetPatchedStateForTesting(): void {
  _isFIPatched = false;
}

type CallbackManagerModule = typeof CallbackManagerModuleV02;

/**
 * @param instrumentationConfig The config for the instrumentation @see {@link InstrumentationConfig}
 * @param traceConfig The FI trace configuration. Can be used to mask or redact sensitive information on spans. @see {@link TraceConfigOptions}
 */
export class LangChainInstrumentation extends InstrumentationBase {
  private fiTracer: FITracer;
  private tracerProvider?: TracerProvider;
  private traceConfig?: TraceConfigOptions;

  constructor({
    instrumentationConfig,
    traceConfig,
    tracerProvider,
  }: {
    /**
     * The config for the instrumentation
     * @see {@link InstrumentationConfig}
     */
    instrumentationConfig?: InstrumentationConfig;
    /**
     * @see {@link TraceConfigOptions}
     */
    traceConfig?: TraceConfigOptions;
    /**
     * An optional custom trace provider to be used for tracing. If not provided, a tracer will be created using the global tracer provider.
     * This is useful if you want to use a non-global tracer provider.
     *
     * @see {@link TracerProvider}
     */
    tracerProvider?: TracerProvider;
  } = {}) {
    super(
      INSTRUMENTATION_NAME,
      VERSION,
      Object.assign({}, instrumentationConfig),
    );
    this.tracerProvider = tracerProvider;
    this.traceConfig = traceConfig;
    this.fiTracer = new FITracer({
      tracer:
        this.tracerProvider?.getTracer(INSTRUMENTATION_NAME, VERSION) ??
        this.tracer,
      traceConfig,
    });
  }

  manuallyInstrument(module: CallbackManagerModule) {
    if (module == null) {

import openai from "openai";
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
import { VERSION } from "./version";
import {
  SemanticConventions,
  FISpanKind,
  MimeType,
  LLMSystem,
  LLMProvider,
} from "@traceai/fi-semantic-conventions";
import {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionCreateParamsBase,
  ChatCompletionMessage,
  ChatCompletionMessageParam,
} from "openai/resources/chat/completions";
import { CompletionCreateParamsBase } from "openai/resources/completions";
import { Stream } from "openai/streaming";
import {
  Completion,
  CreateEmbeddingResponse,
  EmbeddingCreateParams,
} from "openai/resources";
import { assertUnreachable, isString } from "./typeUtils";
import { isTracingSuppressed } from "@opentelemetry/core";

import {
  FITracer,
  safelyJSONStringify,
  TraceConfigOptions,
} from "@traceai/fi-core";
import {
  ResponseCreateParamsBase,
  ResponseStreamEvent,
  Response as ResponseType,
} from "openai/resources/responses/responses";

import {
  consumeResponseStreamEvents,
  getResponsesInputMessagesAttributes,
  getResponsesOutputMessagesAttributes,
  getResponsesUsageAttributes,
} from "./responseAttributes"; 

const MODULE_NAME = "openai";

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
 * Resolves the execution context for the current span
 * If tracing is suppressed, the span is dropped and the current context is returned
 * @param span
 */
function getExecContext(span: Span) {
  const activeContext = context.active();
  const suppressTracing = isTracingSuppressed(activeContext);
  const execContext = suppressTracing
    ? trace.setSpan(context.active(), span)
    : activeContext;
  // Drop the span from the context
  if (suppressTracing) {
    trace.deleteSpan(activeContext);
  }
  return execContext;
}
/**
 * @param instrumentationConfig The config for the instrumentation @see {@link InstrumentationConfig}
 * @param traceConfig The FI trace configuration. Can be used to mask or redact sensitive information on spans. @see {@link TraceConfigOptions}
 */
export class OpenAIInstrumentation extends InstrumentationBase{
  private fiTracer!: FITracer;
  private _traceConfig?: TraceConfigOptions;

  constructor({
    instrumentationConfig,
    traceConfig,
  }: {
    /**
     * The config for the instrumentation
     * @see {@link InstrumentationConfig}
     */
    instrumentationConfig?: InstrumentationConfig;
    /**
     * The fi trace configuration. Can be used to mask or redact sensitive information on spans.
     * @see {@link TraceConfigOptions}
     */
    traceConfig?: TraceConfigOptions;
  } = {}) {
    super(
      "@traceai/fi-instrumentation-openai",
      VERSION,
      Object.assign({}, instrumentationConfig),
    );
    this._traceConfig = traceConfig;
  }

  public override enable(): void {
    super.enable();
    this.fiTracer = new FITracer({ tracer: this.tracer, traceConfig: this._traceConfig });
  }

  protected init() {
    const module = new InstrumentationNodeModuleDefinition(
      "openai",
      ["^4.0.0", "^5.0.0"],
      this.patch.bind(this),
      this.unpatch.bind(this),
    );
    return module;
  }

  /**
   * Manually instruments the OpenAI module. This is needed when the module is not loaded via require (commonjs)
   * @param {openai} module
   */
  manuallyInstrument(module: typeof openai) {
    // diag.debug(`Manually instrumenting ${MODULE_NAME}`);
    this.patch(module);
  }

  /**
   * Patches the OpenAI module
   */
  private patch(
    module: typeof openai & { fiPatched?: boolean },
    moduleVersion?: string,
  ) {
    // diag.debug(`Applying patch for ${MODULE_NAME}@${moduleVersion}`);
    if (module?.fiPatched || _isFIPatched) {
      return module;
    }
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const instrumentation: OpenAIInstrumentation = this;

    // ADDED: Ensure fiTracer uses the most up-to-date tracer from InstrumentationBase
    // The `tracer` property in `InstrumentationBase` (accessed via `instrumentation.tracer`)
    // is initialized as a ProxyTracer and then updated to the actual Tracer.
    // FITracer might have been initialized with the ProxyTracer in `enable()`.
    // Here, we ensure it uses the (potentially updated) Tracer.
    // Accessing FITracer's internal tracer `tracer` for comparison.
    // This assumes FITracer has an internal property named `tracer`.
    if (!(instrumentation.fiTracer as any).tracer || (instrumentation.fiTracer as any).tracer !== instrumentation.tracer) {
        // diag.debug(
        //     `OpenAIInstrumentation.patch: fiTracer's internal tracer (${(instrumentation.fiTracer as any).tracer?.constructor?.name}) ` +
        //     `differs from current base tracer (${instrumentation.tracer?.constructor?.name}) or is not set. Re-initializing fiTracer.`
        // );
        instrumentation.fiTracer = new FITracer({ tracer: instrumentation.tracer, traceConfig: instrumentation._traceConfig });
    } else {
        // diag.debug(
        //     `OpenAIInstrumentation.patch: fiTracer already using current base tracer (${instrumentation.tracer?.constructor?.name}). No re-initialization needed.`
        // );
    }

    // Patch create chat completions
    type ChatCompletionCreateType =
      typeof module.OpenAI.Chat.Completions.prototype.create;

    this._wrap(
      module.OpenAI.Chat.Completions.prototype,
      "create",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (original: ChatCompletionCreateType): any => {
        return function patchedCreate(
          this: unknown,
          ...args: Parameters<ChatCompletionCreateType>
        ) {
          const body = args[0];
          const { messages: _messages, ...invocationParameters } = body;

          // diag.debug("@traceai/openai: ChatCompletion patch CALLED. Starting span...");
          
          // --- ADD LOGS FOR TRACER AND CONTEXT ---
          // const activeContextForSuppressionCheck = context.active();
          // const isSuppressed = isTracingSuppressed(activeContextForSuppressionCheck);
          // diag.debug(`@traceai/openai: Is tracing suppressed? ${isSuppressed}`);
          // diag.debug(`@traceai/openai: this.tracer type: ${instrumentation.tracer?.constructor?.name}`); // Accessing the base tracer
          // diag.debug(`@traceai/openai: this.fiTracer type: ${instrumentation.fiTracer?.constructor?.name}`);
          // --- END ADDED LOGS ---

          const span = instrumentation.fiTracer.startSpan(
            `OpenAI Chat Completions`,
            {
              kind: SpanKind.INTERNAL,
              attributes: {
                [SemanticConventions.FI_SPAN_KIND]:
                  FISpanKind.LLM,
                [SemanticConventions.LLM_MODEL_NAME]: body.model,
                [SemanticConventions.INPUT_VALUE]: JSON.stringify(body),
                [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
                [SemanticConventions.LLM_INVOCATION_PARAMETERS]:
                  JSON.stringify(invocationParameters),
                [SemanticConventions.LLM_PROVIDER]: LLMProvider.OPENAI,
                [SemanticConventions.GEN_AI_OPERATION_NAME]: "chat",
                ...getLLMInputMessagesAttributes(body),
                ...getLLMToolsJSONSchema(body),
                [SemanticConventions.RAW_INPUT]: safelyJSONStringify(body) ?? "",
              },
            },
          );

          // --- ADD LOG ---
          // diag.debug(`@traceai/openai: ChatCompletion span STARTED: ${span.spanContext().spanId}`);

          const execContext = getExecContext(span);
          const execPromise = safeExecuteInTheMiddle<
            ReturnType<ChatCompletionCreateType>
          >(
            () => {
              return context.with(trace.setSpan(execContext, span), () => {
                return original.apply(this, args);
              });
            },
            (error) => {
              if (error) {
                // --- ADD LOG ---
                diag.error(`@traceai/openai: ChatCompletion error in safeExecuteInTheMiddle: ${error}`);
                span.recordException(error);
                span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
                span.end();
                // --- ADD LOG ---
                // diag.debug(`@traceai/openai: ChatCompletion span ENDED due to error: ${span.spanContext().spanId}`);
              }
            },
          );

          const wrappedPromise = execPromise.then((result) => {
            if (isChatCompletionResponse(result)) {
              span.setAttributes({
                [SemanticConventions.OUTPUT_VALUE]: JSON.stringify(result),
                [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
                // Override the model from the value sent by the server
                [SemanticConventions.LLM_MODEL_NAME]: result.model,
                [SemanticConventions.GEN_AI_RESPONSE_MODEL]: result.model,
                [SemanticConventions.GEN_AI_RESPONSE_ID]: result.id,
                [SemanticConventions.GEN_AI_RESPONSE_FINISH_REASONS]: JSON.stringify(
                  result.choices.map((c) => c.finish_reason).filter(Boolean),
                ),
                ...getChatCompletionLLMOutputMessagesAttributes(result),
                ...getUsageAttributes(result),
                [SemanticConventions.RAW_OUTPUT]: safelyJSONStringify(result) ?? "",
              });
              span.setStatus({ code: SpanStatusCode.OK });
              span.end();
            } else {
              const [leftStream, rightStream] = result.tee();
              consumeChatCompletionStreamChunks(rightStream, span);
              result = leftStream;
            }
            return result;
          }).catch((error: Error) => {
            span.recordException(error);
            span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
            span.end();
            throw error;
          });
          return context.bind(execContext, wrappedPromise);
        };
      },
    );

    // Patch create completions
    type CompletionsCreateType =
      typeof module.OpenAI.Completions.prototype.create;

    this._wrap(
      module.OpenAI.Completions.prototype,
      "create",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (original: CompletionsCreateType): any => {
        return function patchedCreate(
          this: unknown,
          ...args: Parameters<CompletionsCreateType>
        ) {
          const body = args[0];
          const { prompt: _prompt, ...invocationParameters } = body;
          const span = instrumentation.fiTracer.startSpan(
            `OpenAI Completions`,
            {
              kind: SpanKind.INTERNAL,
              attributes: {
                [SemanticConventions.FI_SPAN_KIND]:
                  FISpanKind.LLM,
                [SemanticConventions.LLM_MODEL_NAME]: body.model,
                [SemanticConventions.LLM_INVOCATION_PARAMETERS]:
                  JSON.stringify(invocationParameters),
                [SemanticConventions.LLM_PROVIDER]: LLMProvider.OPENAI,
                [SemanticConventions.GEN_AI_OPERATION_NAME]: "text_completion",
                ...getCompletionInputValueAndMimeType(body),
                [SemanticConventions.RAW_INPUT]: safelyJSONStringify(body) ?? "",
              },
            },
          );
          const execContext = getExecContext(span);

          const execPromise = safeExecuteInTheMiddle<
            ReturnType<CompletionsCreateType>
          >(
            () => {
              return context.with(trace.setSpan(execContext, span), () => {
                return original.apply(this, args);
              });
            },
            (error) => {
              // Push the error to the span
              if (error) {
                span.recordException(error);
                span.setStatus({
                  code: SpanStatusCode.ERROR,
                  message: error.message,
                });
                span.end();
              }
            },
          );
          const wrappedPromise = execPromise.then((result) => {
            if (isCompletionResponse(result)) {
              span.setAttributes({
                [SemanticConventions.OUTPUT_VALUE]: JSON.stringify(result),
                [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
                [SemanticConventions.LLM_MODEL_NAME]: result.model,
                [SemanticConventions.GEN_AI_RESPONSE_MODEL]: result.model,
                [SemanticConventions.GEN_AI_RESPONSE_ID]: result.id,
                [SemanticConventions.GEN_AI_RESPONSE_FINISH_REASONS]: JSON.stringify(
                  result.choices.map((c) => c.finish_reason).filter(Boolean),
                ),
                ...getCompletionOutputValueAndMimeType(result),
                ...getUsageAttributes(result),
                [SemanticConventions.RAW_OUTPUT]: safelyJSONStringify(result) ?? "",
              });
              span.setStatus({ code: SpanStatusCode.OK });
              span.end();
            }
            return result;
          }).catch((error: Error) => {
            span.recordException(error);
            span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
            span.end();
            throw error;
          });
          return context.bind(execContext, wrappedPromise);
        };
      },
    );

    // Patch embeddings
    type EmbeddingsCreateType =
      typeof module.OpenAI.Embeddings.prototype.create;
    this._wrap(
      module.OpenAI.Embeddings.prototype,
      "create",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (original: EmbeddingsCreateType): any => {
        return function patchedEmbeddingCreate(
          this: unknown,
          ...args: Parameters<EmbeddingsCreateType>
        ) {
          const body = args[0];
          const { input } = body;
          const isStringInput = typeof input === "string";
          const span = instrumentation.fiTracer.startSpan(`OpenAI Embeddings`, {
            kind: SpanKind.INTERNAL,

import {
  InstrumentationBase,
  InstrumentationConfig,
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
import { isTracingSuppressed } from "@opentelemetry/core";
import {
  FITracer,
  safelyJSONStringify,
  TraceConfigOptions,
} from "@traceai/fi-core";
import { isString } from "./typeUtils";

const MODULE_NAME = "cohere-ai";

/**
 * Flag to check if the Cohere module has been patched
 */
let _isFIPatched = false;

/**
 * Function to check if instrumentation is enabled / disabled
 */
export function isPatched() {
  return _isFIPatched;
}

/**
 * Resolves the execution context for the current span
 */
function getExecContext(span: Span) {
  const activeContext = context.active();
  const suppressTracing = isTracingSuppressed(activeContext);
  const execContext = suppressTracing
    ? trace.setSpan(context.active(), span)
    : activeContext;
  if (suppressTracing) {
    trace.deleteSpan(activeContext);
  }
  return execContext;
}

// Type definitions for Cohere SDK
interface CohereChatRequest {
  model?: string;
  message: string;
  preamble?: string;
  chat_history?: Array<{
    role: string;
    message: string;
  }>;
  temperature?: number;
  max_tokens?: number;
  p?: number;
  k?: number;
  tools?: Array<{
    name: string;
    description?: string;
    parameterDefinitions?: Record<string, unknown>;
  }>;
  [key: string]: unknown;
}

interface CohereChatResponse {
  text: string;
  generation_id?: string;
  chat_history?: Array<{
    role: string;
    message: string;
  }>;
  tool_calls?: Array<{
    name: string;
    parameters: Record<string, unknown>;
  }>;
  meta?: {
    api_version?: { version: string };
    billed_units?: {
      input_tokens?: number;
      output_tokens?: number;
    };
    tokens?: {
      input_tokens?: number;
      output_tokens?: number;
    };
  };
}

interface CohereEmbedRequest {
  texts: string[];
  model?: string;
  input_type?: string;
  embedding_types?: string[];
  truncate?: string;
}

interface CohereEmbedResponse {
  id: string;
  embeddings: number[][] | { float?: number[][] };
  texts: string[];
  meta?: {
    api_version?: { version: string };
    billed_units?: {
      input_tokens?: number;
    };
  };
}

interface CohereRerankRequest {
  query: string;
  documents: string[] | Array<{ text: string }>;
  model?: string;
  top_n?: number;
  return_documents?: boolean;
}

interface CohereRerankResponse {
  id: string;
  results: Array<{
    index: number;
    relevance_score: number;
    document?: { text: string };
  }>;
  meta?: {
    api_version?: { version: string };
    billed_units?: {
      search_units?: number;
    };
  };
}

/**
 * @param instrumentationConfig The config for the instrumentation
 * @param traceConfig The FI trace configuration for masking sensitive information
 */
export class CohereInstrumentation extends InstrumentationBase {
  private fiTracer!: FITracer;
  private _traceConfig?: TraceConfigOptions;
  public readonly instrumentationName = "@traceai/cohere";

  constructor({
    instrumentationConfig,
    traceConfig,
  }: {
    instrumentationConfig?: InstrumentationConfig;
    traceConfig?: TraceConfigOptions;
  } = {}) {
    super(
      "@traceai/cohere",
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
      "cohere-ai",
      [">=7.0.0"],
      this.patch.bind(this),
      this.unpatch.bind(this),
    );
    return module;
  }

  /**
   * Manually instruments the Cohere module
   */
  manuallyInstrument(module: Record<string, unknown>) {
    this.patch(module);
  }

  /**
   * Patches the Cohere module
   */
  private patch(
    module: Record<string, unknown> & { fiPatched?: boolean },
    moduleVersion?: string,
  ) {
    if (module?.fiPatched || _isFIPatched) {
      return module;
    }

    const instrumentation: CohereInstrumentation = this;

    // Ensure fiTracer uses the most up-to-date tracer
    if (!(instrumentation.fiTracer as unknown as Record<string, unknown>)?.tracer ||
        (instrumentation.fiTracer as unknown as Record<string, unknown>)?.tracer !== instrumentation.tracer) {
      instrumentation.fiTracer = new FITracer({
        tracer: instrumentation.tracer,
        traceConfig: instrumentation._traceConfig
      });
    }

    // Patch CohereClient class
    const CohereClient = module.CohereClient as {
      prototype?: Record<string, unknown>;
    } | undefined;

    if (CohereClient?.prototype) {
      // Patch chat method
      if (typeof CohereClient.prototype.chat === 'function') {
        this._wrap(
          CohereClient.prototype,
          "chat",
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (original: any): any => {
            return function patchedChat(
              this: unknown,
              ...args: [CohereChatRequest]
            ) {
              const request = args[0];
              const modelName = request.model || 'command';
              const { message, chat_history, preamble, ...invocationParameters } = request;

              const span = instrumentation.fiTracer.startSpan(
                `Cohere Chat`,
                {
                  kind: SpanKind.INTERNAL,
                  attributes: {
                    [SemanticConventions.FI_SPAN_KIND]: FISpanKind.LLM,
                    [SemanticConventions.LLM_MODEL_NAME]: modelName,
                    [SemanticConventions.INPUT_VALUE]: request.message,
                    [SemanticConventions.INPUT_MIME_TYPE]: MimeType.TEXT,
                    [SemanticConventions.LLM_INVOCATION_PARAMETERS]:
                      JSON.stringify(invocationParameters),
                    [SemanticConventions.LLM_PROVIDER]: LLMProvider.COHERE,
                    [SemanticConventions.GEN_AI_OPERATION_NAME]: "chat",
                    ...getChatInputMessagesAttributes(request),
                    ...getChatToolsAttributes(request),
                    [SemanticConventions.RAW_INPUT]: safelyJSONStringify(request) ?? "",
                  },
                },
              );

              const execContext = getExecContext(span);
              const execPromise = safeExecuteInTheMiddle(
                () => {
                  return context.with(trace.setSpan(execContext, span), () => {
                    return original.apply(this, args);
                  });
                },
                (error) => {
                  if (error) {
                    span.recordException(error);
                    span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
                    span.end();
                  }
                },
              );

              const wrappedPromise = (execPromise as Promise<CohereChatResponse>).then((result) => {
                span.setAttributes({
                  [SemanticConventions.OUTPUT_VALUE]: result.text,
                  [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.TEXT,
                  ...getChatOutputMessagesAttributes(result),
                  ...getChatUsageAttributes(result),
                  [SemanticConventions.RAW_OUTPUT]: safelyJSONStringify(result) ?? "",
                });
                span.setStatus({ code: SpanStatusCode.OK });
                span.end();
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
      }

      // Patch embed method
      if (typeof CohereClient.prototype.embed === 'function') {
        this._wrap(
          CohereClient.prototype,
          "embed",
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (original: any): any => {
            return function patchedEmbed(
              this: unknown,
              ...args: [CohereEmbedRequest]
            ) {
              const request = args[0];
              const modelName = request.model || 'embed-english-v3.0';

              const span = instrumentation.fiTracer.startSpan(`Cohere Embed`, {
                kind: SpanKind.INTERNAL,
                attributes: {
                  [SemanticConventions.FI_SPAN_KIND]: FISpanKind.EMBEDDING,
                  [SemanticConventions.EMBEDDING_MODEL_NAME]: modelName,
                  [SemanticConventions.INPUT_VALUE]: JSON.stringify(request.texts),
                  [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
                  [SemanticConventions.LLM_PROVIDER]: LLMProvider.COHERE,
                  [SemanticConventions.GEN_AI_OPERATION_NAME]: "embeddings",
                  ...getEmbedTextAttributes(request),
                  [SemanticConventions.RAW_INPUT]: safelyJSONStringify(request) ?? "",
                },
              });

              const execContext = getExecContext(span);
              const execPromise = safeExecuteInTheMiddle(
                () => {
                  return context.with(trace.setSpan(execContext, span), () => {
                    return original.apply(this, args);
                  });
                },
                (error) => {
                  if (error) {
                    span.recordException(error);
                    span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
                    span.end();
                  }
                },
              );

              const wrappedPromise = (execPromise as Promise<CohereEmbedResponse>).then((result) => {
                span.setAttributes({
                  ...getEmbedVectorAttributes(result),
                  [SemanticConventions.RAW_OUTPUT]: safelyJSONStringify(result) ?? "",
                });
                span.setStatus({ code: SpanStatusCode.OK });
                span.end();
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
      }

      // Patch rerank method
      if (typeof CohereClient.prototype.rerank === 'function') {
        this._wrap(
          CohereClient.prototype,
          "rerank",
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (original: any): any => {
            return function patchedRerank(
              this: unknown,
              ...args: [CohereRerankRequest]
            ) {
              const request = args[0];
              const modelName = request.model || 'rerank-english-v3.0';

              const span = instrumentation.fiTracer.startSpan(`Cohere Rerank`, {
                kind: SpanKind.INTERNAL,
                attributes: {
                  [SemanticConventions.FI_SPAN_KIND]: FISpanKind.LLM,
                  [SemanticConventions.LLM_MODEL_NAME]: modelName,
                  [SemanticConventions.INPUT_VALUE]: request.query,
                  [SemanticConventions.INPUT_MIME_TYPE]: MimeType.TEXT,
                  [SemanticConventions.LLM_PROVIDER]: LLMProvider.COHERE,
                  [SemanticConventions.GEN_AI_OPERATION_NAME]: "rerank",
                  "cohere.rerank.query": request.query,
                  "cohere.rerank.documents_count": request.documents.length,
                  "cohere.rerank.top_n": request.top_n,
                  [SemanticConventions.RAW_INPUT]: safelyJSONStringify(request) ?? "",
                },
              });

              const execContext = getExecContext(span);
              const execPromise = safeExecuteInTheMiddle(
                () => {
                  return context.with(trace.setSpan(execContext, span), () => {
                    return original.apply(this, args);
                  });
                },
                (error) => {
                  if (error) {
                    span.recordException(error);
                    span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
                    span.end();
                  }
                },
              );

              const wrappedPromise = (execPromise as Promise<CohereRerankResponse>).then((result) => {
                span.setAttributes({
                  "cohere.rerank.results_count": result.results.length,
                  ...getRerankResultsAttributes(result),
                  [SemanticConventions.RAW_OUTPUT]: safelyJSONStringify(result) ?? "",
                });

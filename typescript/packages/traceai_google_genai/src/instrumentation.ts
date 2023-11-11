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

const MODULE_NAME = "@google/generative-ai";

/**
 * Flag to check if the Google GenAI module has been patched
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

// Type definitions for Google GenAI SDK
interface GenerateContentRequest {
  contents: Array<{
    role?: string;
    parts: Array<{
      text?: string;
      inlineData?: {
        mimeType: string;
        data: string;
      };
    }>;
  }>;
  generationConfig?: {
    temperature?: number;
    topP?: number;
    topK?: number;
    maxOutputTokens?: number;
    stopSequences?: string[];
  };
  safetySettings?: Array<{
    category: string;
    threshold: string;
  }>;
  tools?: Array<{
    functionDeclarations?: Array<{
      name: string;
      description?: string;
      parameters?: Record<string, unknown>;
    }>;
  }>;
  systemInstruction?: {
    parts: Array<{ text: string }>;
  };
}

interface GenerateContentResponse {
  response: {
    text: () => string;
    candidates?: Array<{
      content: {
        role: string;
        parts: Array<{
          text?: string;
          functionCall?: {
            name: string;
            args: Record<string, unknown>;
          };
        }>;
      };
      finishReason?: string;
      safetyRatings?: Array<{
        category: string;
        probability: string;
      }>;
    }>;
    usageMetadata?: {
      promptTokenCount: number;
      candidatesTokenCount: number;
      totalTokenCount: number;
    };
  };
}

interface EmbedContentRequest {
  content: {
    parts: Array<{ text: string }>;
  };
  taskType?: string;
  title?: string;
}

interface EmbedContentResponse {
  embedding: {
    values: number[];
  };
}

interface BatchEmbedContentsRequest {
  requests: EmbedContentRequest[];
}

interface BatchEmbedContentsResponse {
  embeddings: Array<{
    values: number[];
  }>;
}

/**
 * @param instrumentationConfig The config for the instrumentation
 * @param traceConfig The FI trace configuration for masking sensitive information
 */
export class GoogleGenAIInstrumentation extends InstrumentationBase {
  private fiTracer!: FITracer;
  private _traceConfig?: TraceConfigOptions;
  public readonly instrumentationName = "@traceai/google-genai";

  constructor({
    instrumentationConfig,
    traceConfig,
  }: {
    instrumentationConfig?: InstrumentationConfig;
    traceConfig?: TraceConfigOptions;
  } = {}) {
    super(
      "@traceai/google-genai",
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
      "@google/generative-ai",
      [">=0.1.0"],
      this.patch.bind(this),
      this.unpatch.bind(this),
    );
    return module;
  }

  /**
   * Manually instruments the Google GenAI module
   */
  manuallyInstrument(module: Record<string, unknown>) {
    this.patch(module);
  }

  /**
   * Patches the Google GenAI module
   */
  private patch(
    module: Record<string, unknown> & { fiPatched?: boolean },
    moduleVersion?: string,
  ) {
    if (module?.fiPatched || _isFIPatched) {
      return module;
    }

    const instrumentation: GoogleGenAIInstrumentation = this;

    // Ensure fiTracer uses the most up-to-date tracer
    if (!(instrumentation.fiTracer as unknown as Record<string, unknown>)?.tracer ||
        (instrumentation.fiTracer as unknown as Record<string, unknown>)?.tracer !== instrumentation.tracer) {
      instrumentation.fiTracer = new FITracer({
        tracer: instrumentation.tracer,
        traceConfig: instrumentation._traceConfig
      });
    }

    // Patch GenerativeModel class
    const GenerativeModel = module.GenerativeModel as {
      prototype?: Record<string, unknown>;
    } | undefined;

    if (GenerativeModel?.prototype) {
      // Patch generateContent method
      if (typeof GenerativeModel.prototype.generateContent === 'function') {
        this._wrap(
          GenerativeModel.prototype,
          "generateContent",
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (original: any): any => {
            return function patchedGenerateContent(
              this: { model?: string },
              ...args: [GenerateContentRequest | string]
            ) {
              const request = typeof args[0] === 'string'
                ? { contents: [{ parts: [{ text: args[0] }] }] }
                : args[0];

              const modelName = this.model || 'unknown';

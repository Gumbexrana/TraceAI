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

const MODULE_NAME = "together-ai";

/**
 * Flag to check if the together module has been patched
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

// Type definitions for Together AI SDK
interface TogetherMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  tool_calls?: TogetherToolCall[];
  tool_call_id?: string;
}

interface TogetherToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

interface TogetherTool {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

interface TogetherChatCompletionRequest {
  model: string;
  messages: TogetherMessage[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  repetition_penalty?: number;
  stop?: string | string[];
  stream?: boolean;
  tools?: TogetherTool[];
  tool_choice?: string | { type: string; function: { name: string } };
  response_format?: { type: string };
  [key: string]: unknown;
}

interface TogetherChatCompletionChoice {
  index: number;
  message: TogetherMessage;
  finish_reason: string;
  delta?: {
    content?: string;
    role?: string;
    tool_calls?: TogetherToolCall[];
  };
}

interface TogetherChatCompletion {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: TogetherChatCompletionChoice[];
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface TogetherCompletionRequest {
  model: string;
  prompt: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  repetition_penalty?: number;
  stop?: string | string[];
  stream?: boolean;
  [key: string]: unknown;
}

interface TogetherCompletionChoice {
  text: string;
  index: number;
  finish_reason: string;
}

interface TogetherCompletion {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: TogetherCompletionChoice[];
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface TogetherEmbeddingRequest {
  model: string;
  input: string | string[];
}

interface TogetherEmbeddingData {
  object: string;
  embedding: number[];
  index: number;
}

interface TogetherEmbeddingResponse {
  object: string;
  data: TogetherEmbeddingData[];
  model: string;
  usage?: {
    prompt_tokens: number;
    total_tokens: number;
  };
}

interface TogetherStream<T> {
  [Symbol.asyncIterator](): AsyncIterator<T>;
}

/**
 * @param instrumentationConfig The config for the instrumentation
 * @param traceConfig The FI trace configuration for masking sensitive information
 */
export class TogetherInstrumentation extends InstrumentationBase {
  private fiTracer!: FITracer;
  private _traceConfig?: TraceConfigOptions;

  constructor({
    instrumentationConfig,
    traceConfig,
  }: {
    instrumentationConfig?: InstrumentationConfig;
    traceConfig?: TraceConfigOptions;
  } = {}) {
    super(
      "@traceai/together",
      VERSION,
      Object.assign({}, instrumentationConfig),
    );
    this._traceConfig = traceConfig;
  }

  public override enable(): void {
    super.enable();
    this.fiTracer = new FITracer({ tracer: this.tracer, traceConfig: this._traceConfig });
  }

  public readonly instrumentationName = "@traceai/together";

  protected init() {
    const module = new InstrumentationNodeModuleDefinition(
      "together-ai",
      [">=0.5.0"],
      this.patch.bind(this),
      this.unpatch.bind(this),
    );
    return module;
  }

  /**
   * Manually instruments the Together module
   */
  manuallyInstrument(module: Record<string, unknown>) {
    this.patch(module);
  }

  /**
   * Patches the Together module
   */
  private patch(
    module: Record<string, unknown> & { fiPatched?: boolean },
    moduleVersion?: string,
  ) {
    if (module?.fiPatched || _isFIPatched) {
      return module;
    }

    const instrumentation: TogetherInstrumentation = this;

    // Ensure fiTracer uses the most up-to-date tracer
    if (!(instrumentation.fiTracer as unknown as Record<string, unknown>)?.tracer ||
        (instrumentation.fiTracer as unknown as Record<string, unknown>)?.tracer !== instrumentation.tracer) {
      instrumentation.fiTracer = new FITracer({
        tracer: instrumentation.tracer,
        traceConfig: instrumentation._traceConfig
      });
    }

    // Patch the Together class
    const Together = module.Together as {
      prototype: {
        chat?: { completions?: { create?: unknown } };
        completions?: { create?: unknown };
        embeddings?: { create?: unknown };
      };
    } | undefined;

    if (Together?.prototype) {
      // Patch chat.completions.create
      if (Together.prototype.chat?.completions && typeof (Together.prototype.chat.completions as Record<string, unknown>).create === "function") {
        this._wrap(
          Together.prototype.chat.completions as Record<string, unknown>,
          "create",
          this.createChatCompletionWrapper(instrumentation),
        );
      }

      // Patch completions.create (legacy)
      if (Together.prototype.completions && typeof (Together.prototype.completions as Record<string, unknown>).create === "function") {
        this._wrap(
          Together.prototype.completions as Record<string, unknown>,
          "create",
          this.createCompletionWrapper(instrumentation),
        );
      }

      // Patch embeddings.create
      if (Together.prototype.embeddings && typeof (Together.prototype.embeddings as Record<string, unknown>).create === "function") {
        this._wrap(
          Together.prototype.embeddings as Record<string, unknown>,
          "create",
          this.createEmbeddingWrapper(instrumentation),
        );
      }
    }

    _isFIPatched = true;
    try {

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

const MODULE_NAME = "openai";

/**
 * Flag to check if the Fireworks module has been patched
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

// Type definitions for OpenAI-compatible API
interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | null;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}

interface ToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

interface Tool {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

interface ChatCompletionRequest {
  model: string;
  messages: ChatMessage[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  stop?: string | string[];
  stream?: boolean;
  tools?: Tool[];
  tool_choice?: string | { type: string; function: { name: string } };
  [key: string]: unknown;
}

interface ChatCompletionChoice {
  index: number;
  message: ChatMessage;
  finish_reason: string;
  delta?: {
    content?: string;
    role?: string;
    tool_calls?: ToolCall[];
  };
}

interface ChatCompletion {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface CompletionRequest {
  model: string;
  prompt: string | string[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop?: string | string[];
  stream?: boolean;
  [key: string]: unknown;
}

interface CompletionChoice {
  text: string;
  index: number;
  finish_reason: string;
}

interface Completion {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: CompletionChoice[];
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface EmbeddingRequest {
  model: string;
  input: string | string[];
}

interface EmbeddingData {
  object: string;
  embedding: number[];
  index: number;
}

interface EmbeddingResponse {
  object: string;
  data: EmbeddingData[];
  model: string;
  usage?: {
    prompt_tokens: number;
    total_tokens: number;
  };
}

interface StreamResponse<T> {
  [Symbol.asyncIterator](): AsyncIterator<T>;
}

/**
 * @param instrumentationConfig The config for the instrumentation
 * @param traceConfig The FI trace configuration for masking sensitive information
 */
export class FireworksInstrumentation extends InstrumentationBase {
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
      "@traceai/fireworks",
      VERSION,
      Object.assign({}, instrumentationConfig),
    );
    this._traceConfig = traceConfig;
  }

  public override enable(): void {
    super.enable();
    this.fiTracer = new FITracer({ tracer: this.tracer, traceConfig: this._traceConfig });
  }

  public readonly instrumentationName = "@traceai/fireworks";

  protected init() {
    const module = new InstrumentationNodeModuleDefinition(
      "openai",
      [">=4.0.0"],
      this.patch.bind(this),
      this.unpatch.bind(this),
    );
    return module;
  }

  /**
   * Manually instruments the OpenAI module for Fireworks AI
   */
  manuallyInstrument(module: Record<string, unknown>) {
    this.patch(module);
  }

  /**
   * Checks if a request is targeting Fireworks AI API
   */
  private isFireworksRequest(client: unknown): boolean {
    const baseURL = (client as { baseURL?: string })?.baseURL;
    if (!baseURL) return false;
    return baseURL.includes("fireworks") || baseURL.includes("api.fireworks.ai");
  }

  /**
   * Patches the OpenAI module for Fireworks AI
   */
  private patch(
    module: Record<string, unknown> & { fiFireworksPatched?: boolean },
    moduleVersion?: string,
  ) {
    if (module?.fiFireworksPatched || _isFIPatched) {
      return module;
    }

    const instrumentation: FireworksInstrumentation = this;

    // Ensure fiTracer uses the most up-to-date tracer
    if (!(instrumentation.fiTracer as unknown as Record<string, unknown>)?.tracer ||
        (instrumentation.fiTracer as unknown as Record<string, unknown>)?.tracer !== instrumentation.tracer) {
      instrumentation.fiTracer = new FITracer({
        tracer: instrumentation.tracer,
        traceConfig: instrumentation._traceConfig
      });
    }

    // Patch the OpenAI class
    const OpenAI = module.OpenAI as {
      prototype: {
        chat?: { completions?: { create?: unknown } };
        completions?: { create?: unknown };
        embeddings?: { create?: unknown };
      };
    } | undefined;

    if (OpenAI?.prototype) {
      // Patch chat.completions.create
      if (OpenAI.prototype.chat?.completions && typeof (OpenAI.prototype.chat.completions as Record<string, unknown>).create === "function") {
        this._wrap(
          OpenAI.prototype.chat.completions as Record<string, unknown>,
          "create",
          this.createChatCompletionWrapper(instrumentation),
        );
      }

      // Patch completions.create
      if (OpenAI.prototype.completions && typeof (OpenAI.prototype.completions as Record<string, unknown>).create === "function") {
        this._wrap(
          OpenAI.prototype.completions as Record<string, unknown>,
          "create",
          this.createCompletionWrapper(instrumentation),
        );
      }

      // Patch embeddings.create
      if (OpenAI.prototype.embeddings && typeof (OpenAI.prototype.embeddings as Record<string, unknown>).create === "function") {
        this._wrap(
          OpenAI.prototype.embeddings as Record<string, unknown>,
          "create",
          this.createEmbeddingWrapper(instrumentation),
        );
      }
    }

    _isFIPatched = true;
    try {
      module.fiFireworksPatched = true;
    } catch (e) {
      diag.warn(`Failed to set fiFireworksPatched flag on module '${MODULE_NAME}'. Error: ${e}`);
    }

    return module;

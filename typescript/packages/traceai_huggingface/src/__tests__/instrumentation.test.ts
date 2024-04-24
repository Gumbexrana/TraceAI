import { HuggingFaceInstrumentation, isPatched } from "../instrumentation";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { SemanticConventions, FISpanKind, MimeType, LLMSystem, LLMProvider } from "@traceai/fi-semantic-conventions";

describe("HuggingFaceInstrumentation", () => {
  let instrumentation: HuggingFaceInstrumentation;
  let memoryExporter: InMemorySpanExporter;
  let provider: NodeTracerProvider;

  beforeEach(() => {
    memoryExporter = new InMemorySpanExporter();
    provider = new NodeTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(memoryExporter)],
    });
    provider.register();

    instrumentation = new HuggingFaceInstrumentation();
    instrumentation.setTracerProvider(provider);
    instrumentation.enable();
  });

  afterEach(async () => {
    memoryExporter.reset();
    await provider.shutdown();
  });

  describe("constructor", () => {
    it("should create instrumentation with default config", () => {
      const inst = new HuggingFaceInstrumentation();
      expect(inst.instrumentationName).toBe("@traceai/huggingface");
    });

    it("should create instrumentation with custom config", () => {
      const inst = new HuggingFaceInstrumentation({
        instrumentationConfig: { enabled: true },
        traceConfig: { hideInputs: true },
      });
      expect(inst.instrumentationName).toBe("@traceai/huggingface");
    });
  });

  describe("isPatched", () => {
    it("should return boolean", () => {
      expect(typeof isPatched()).toBe("boolean");
    });
  });

  describe("manuallyInstrument", () => {
    it("should patch a mock HuggingFace module", () => {
      const mockModule = {
        HfInference: class HfInference {
          textGeneration = jest.fn();
          chatCompletion = jest.fn();
          featureExtraction = jest.fn();
          summarization = jest.fn();
          translation = jest.fn();
          questionAnswering = jest.fn();
        },
      };

      // Set up prototype methods
      (mockModule.HfInference.prototype as unknown as Record<string, unknown>).textGeneration = jest.fn();
      (mockModule.HfInference.prototype as unknown as Record<string, unknown>).chatCompletion = jest.fn();
      (mockModule.HfInference.prototype as unknown as Record<string, unknown>).chatCompletionStream = jest.fn();
      (mockModule.HfInference.prototype as unknown as Record<string, unknown>).featureExtraction = jest.fn();
      (mockModule.HfInference.prototype as unknown as Record<string, unknown>).summarization = jest.fn();
      (mockModule.HfInference.prototype as unknown as Record<string, unknown>).translation = jest.fn();
      (mockModule.HfInference.prototype as unknown as Record<string, unknown>).questionAnswering = jest.fn();

      instrumentation.manuallyInstrument(mockModule as unknown as Record<string, unknown>);
      expect(isPatched()).toBe(true);
    });
  });

  describe("textGeneration wrapper", () => {
    let mockOriginal: jest.Mock;
    let patchedFn: (args: Record<string, unknown>) => Promise<unknown>;

    beforeEach(() => {
      mockOriginal = jest.fn().mockResolvedValue({
        generated_text: "The quick brown fox jumps over the lazy dog.",
      });

      const wrapperFactory = (instrumentation as unknown as {
        createTextGenerationWrapper: (inst: HuggingFaceInstrumentation) => (original: jest.Mock) => typeof patchedFn;
      }).createTextGenerationWrapper(instrumentation);
      patchedFn = wrapperFactory(mockOriginal);
    });

    it("should create a span for text generation", async () => {
      const args = {
        model: "gpt2",
        inputs: "The quick brown",
      };

      await patchedFn(args);

      const spans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBe(1);
      expect(spans[0].name).toBe("HuggingFace Text Generation");
    });

    it("should set correct attributes", async () => {
      const args = {
        model: "gpt2",
        inputs: "The quick brown",
        parameters: {
          max_new_tokens: 50,
          temperature: 0.7,
        },
      };

      await patchedFn(args);

      const spans = memoryExporter.getFinishedSpans();
      const span = spans[0];

      expect(span.attributes[SemanticConventions.FI_SPAN_KIND]).toBe(FISpanKind.LLM);
      expect(span.attributes[SemanticConventions.LLM_MODEL_NAME]).toBe("gpt2");
      expect(span.attributes[SemanticConventions.LLM_PROVIDER]).toBe(LLMProvider.HUGGINGFACE);
      expect(span.attributes[SemanticConventions.INPUT_VALUE]).toBe("The quick brown");
      expect(span.attributes[SemanticConventions.INPUT_MIME_TYPE]).toBe(MimeType.TEXT);
    });

    it("should capture output", async () => {
      const args = {
        model: "gpt2",
        inputs: "The quick brown",
      };

      await patchedFn(args);

      const spans = memoryExporter.getFinishedSpans();
      const span = spans[0];

      expect(span.attributes[SemanticConventions.OUTPUT_VALUE]).toBe("The quick brown fox jumps over the lazy dog.");
      expect(span.attributes[SemanticConventions.OUTPUT_MIME_TYPE]).toBe(MimeType.TEXT);
    });

    it("should capture prompt", async () => {
      const args = {
        model: "gpt2",
        inputs: "The quick brown",
      };

      await patchedFn(args);

      const spans = memoryExporter.getFinishedSpans();
      const span = spans[0];

      expect(span.attributes[`${SemanticConventions.LLM_PROMPTS}.0`]).toBe("The quick brown");
    });

    it("should handle errors", async () => {
      const error = new Error("Text generation error");
      mockOriginal.mockRejectedValueOnce(error);

      const args = {
        model: "gpt2",
        inputs: "The quick brown",
      };

      await expect(patchedFn(args)).rejects.toThrow("Text generation error");

      const spans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBe(1);
      expect(spans[0].status.code).toBe(2);
    });

    it("should handle unknown model", async () => {
      const args = {
        inputs: "The quick brown",
      };

      await patchedFn(args);

      const spans = memoryExporter.getFinishedSpans();
      const span = spans[0];

      expect(span.attributes[SemanticConventions.LLM_MODEL_NAME]).toBe("unknown");
    });
  });

  describe("chatCompletion wrapper", () => {
    let mockOriginal: jest.Mock;
    let patchedFn: (args: Record<string, unknown>) => Promise<unknown>;

    beforeEach(() => {
      mockOriginal = jest.fn().mockResolvedValue({
        id: "chatcmpl-123",
        created: 1677652288,
        model: "meta-llama/Llama-2-7b-chat-hf",
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: "Hello! How can I help you today?",
            },
            finish_reason: "stop",
          },
        ],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 8,
          total_tokens: 18,
        },
      });

      const wrapperFactory = (instrumentation as unknown as {
        createChatCompletionWrapper: (inst: HuggingFaceInstrumentation) => (original: jest.Mock) => typeof patchedFn;
      }).createChatCompletionWrapper(instrumentation);
      patchedFn = wrapperFactory(mockOriginal);
    });

    it("should create a span for chat completion", async () => {
      const args = {
        model: "meta-llama/Llama-2-7b-chat-hf",
        messages: [
          { role: "user", content: "Hello!" },
        ],
      };

      await patchedFn(args);

      const spans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBe(1);
      expect(spans[0].name).toBe("HuggingFace Chat Completion");
    });

    it("should set correct attributes", async () => {
      const args = {
        model: "meta-llama/Llama-2-7b-chat-hf",
        messages: [
          { role: "system", content: "You are helpful." },
          { role: "user", content: "Hello!" },
        ],
        max_tokens: 100,
        temperature: 0.7,
      };

      await patchedFn(args);

      const spans = memoryExporter.getFinishedSpans();
      const span = spans[0];

      expect(span.attributes[SemanticConventions.FI_SPAN_KIND]).toBe(FISpanKind.LLM);
      expect(span.attributes[SemanticConventions.LLM_MODEL_NAME]).toBe("meta-llama/Llama-2-7b-chat-hf");
      expect(span.attributes[SemanticConventions.LLM_PROVIDER]).toBe(LLMProvider.HUGGINGFACE);
      expect(span.attributes[SemanticConventions.INPUT_MIME_TYPE]).toBe(MimeType.JSON);
    });

    it("should capture input messages", async () => {
      const args = {
        model: "meta-llama/Llama-2-7b-chat-hf",
        messages: [
          { role: "system", content: "You are helpful." },
          { role: "user", content: "Hello!" },
        ],
      };

      await patchedFn(args);

      const spans = memoryExporter.getFinishedSpans();
      const span = spans[0];

      const inputMessages = JSON.parse(span.attributes[SemanticConventions.LLM_INPUT_MESSAGES] as string);
      expect(inputMessages[0].role).toBe("system");
      expect(inputMessages[0].content).toBe("You are helpful.");
      expect(inputMessages[1].role).toBe("user");
      expect(inputMessages[1].content).toBe("Hello!");
    });

    it("should capture output messages", async () => {
      const args = {
        model: "meta-llama/Llama-2-7b-chat-hf",
        messages: [{ role: "user", content: "Hello!" }],
      };

      await patchedFn(args);

      const spans = memoryExporter.getFinishedSpans();
      const span = spans[0];

      const outputMessages = JSON.parse(span.attributes[SemanticConventions.LLM_OUTPUT_MESSAGES] as string);
      expect(outputMessages[0].role).toBe("assistant");
      expect(outputMessages[0].content).toBe("Hello! How can I help you today?");
    });

    it("should capture token usage", async () => {
      const args = {
        model: "meta-llama/Llama-2-7b-chat-hf",
        messages: [{ role: "user", content: "Hello!" }],
      };

      await patchedFn(args);

      const spans = memoryExporter.getFinishedSpans();
      const span = spans[0];

      expect(span.attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT]).toBe(10);
      expect(span.attributes[SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]).toBe(8);
      expect(span.attributes[SemanticConventions.LLM_TOKEN_COUNT_TOTAL]).toBe(18);
    });

    it("should handle errors", async () => {
      const error = new Error("Chat completion error");
      mockOriginal.mockRejectedValueOnce(error);

      const args = {
        model: "meta-llama/Llama-2-7b-chat-hf",
        messages: [{ role: "user", content: "Hello!" }],
      };

      await expect(patchedFn(args)).rejects.toThrow("Chat completion error");

      const spans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBe(1);
      expect(spans[0].status.code).toBe(2);
    });
  });

  describe("chatCompletionStream wrapper", () => {
    let mockOriginal: jest.Mock;
    let patchedFn: (args: Record<string, unknown>) => Promise<unknown>;

    beforeEach(() => {

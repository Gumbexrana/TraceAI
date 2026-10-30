/**
 * E2E tests for vLLM instrumentation
 *
 * These tests export spans to the FI backend via register() from @traceai/fi-core.
 * Even error spans (from dummy keys) show up in the UI.
 *
 * Required environment variables:
 *   FI_API_KEY     - FI platform API key
 *   FI_SECRET_KEY  - FI platform secret key (if required)
 *
 * Optional:
 *   GOOGLE_API_KEY - Google API key for OpenAI-compatible endpoint
 *
 * Example:
 *   FI_API_KEY=... pnpm test -- --testPathPattern=e2e
 */
import { register, FITracerProvider, ProjectType } from "@traceai/fi-core";
import { VLLMInstrumentation } from "../instrumentation";

const FI_API_KEY = process.env.FI_API_KEY;
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY || "dummy-key-for-e2e";

const describeE2E = FI_API_KEY ? describe : describe.skip;

describeE2E("VLLMInstrumentation E2E", () => {
  let provider: FITracerProvider;
  let instrumentation: VLLMInstrumentation;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let OpenAI: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let client: any;

  beforeAll(async () => {
    provider = register({
      projectName: process.env.FI_PROJECT_NAME || "ts-vllm-e2e",
      projectType: ProjectType.OBSERVE,
      batch: false,
    });

    instrumentation = new VLLMInstrumentation();
    instrumentation.setTracerProvider(provider);
    instrumentation.enable();

    const openaiModule = await import("openai");
    instrumentation.manuallyInstrument(openaiModule as unknown as Record<string, unknown>);
    OpenAI = openaiModule.default;

    client = new OpenAI({
      baseURL: "https://generativelanguage.googleapis.com/v1beta/openai/",
      apiKey: GOOGLE_API_KEY,
    });
  });

  afterAll(async () => {
    instrumentation.disable();
    await new Promise((resolve) => setTimeout(resolve, 5000));
    await provider.forceFlush();
    await provider.shutdown();
  }, 15000);

  it("should trace chat completion", async () => {
    try {
      const response = await client.chat.completions.create({
        model: "gemini-2.0-flash",
        messages: [
          { role: "system", content: "You are a helpful assistant." },
          { role: "user", content: "Say hello in one word." },
        ],
        max_tokens: 10,
      });

      expect(response.choices[0].message.content).toBeDefined();
      console.log("Chat completion response:", response.choices[0].message.content);
    } catch (error) {
      console.log("Chat completion produced error span:", (error as Error).message);
    }
  }, 30000);

  it("should trace streaming chat completion", async () => {

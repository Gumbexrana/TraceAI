/**
 * E2E tests for Cohere instrumentation
 *
 * These tests export spans to the FI backend via register() from @traceai/fi-core.
 * Even error spans (from dummy keys) appear in the UI.
 *
 * Required environment variables:
 *   FI_API_KEY     - FI platform API key
 *
 * Example:
 *   FI_API_KEY=... pnpm test -- --testPathPattern=e2e
 */
import { register, FITracerProvider, ProjectType } from "@traceai/fi-core";
import { CohereInstrumentation } from "../instrumentation";

const FI_API_KEY = process.env.FI_API_KEY;

const describeE2E = FI_API_KEY ? describe : describe.skip;

describeE2E("Cohere E2E Tests", () => {
  let provider: FITracerProvider;
  let instrumentation: CohereInstrumentation;
  let CohereClient: typeof import("cohere-ai").CohereClient;
  let client: InstanceType<typeof CohereClient>;

  beforeAll(async () => {
    provider = register({
      projectName: process.env.FI_PROJECT_NAME || "ts-cohere-e2e",
      projectType: ProjectType.OBSERVE,
      batch: false,
    });

    instrumentation = new CohereInstrumentation();
    instrumentation.setTracerProvider(provider);
    instrumentation.enable();

    const cohereModule = await import("cohere-ai");
    instrumentation.manuallyInstrument(cohereModule as unknown as Record<string, unknown>);
    CohereClient = cohereModule.CohereClient;
    client = new CohereClient({ token: process.env.COHERE_API_KEY || "dummy-key-for-e2e" });
  });

  afterAll(async () => {
    instrumentation.disable();
    await new Promise((resolve) => setTimeout(resolve, 5000));
    await provider.forceFlush();
    await provider.shutdown();
  }, 15000);

  describe("Chat", () => {
    it("should complete a basic chat request", async () => {
      try {
        const response = await client.chat({
          message: "What is 2 + 2? Answer with just the number.",
          model: "command-r-08-2024",
        });
        console.log("Chat response:", response.text);
      } catch (error) {
        console.log("Chat errored (span still exported):", (error as Error).message);
      }
    }, 30000);

    it("should handle preamble (system prompt)", async () => {
      try {
        const response = await client.chat({
          message: "Say hello",
          model: "command-r-08-2024",
          preamble: "You are a helpful assistant. Always respond with exactly one word.",
        });
        console.log("Preamble response:", response.text);
      } catch (error) {
        console.log("Preamble errored (span still exported):", (error as Error).message);
      }
    }, 30000);

    it("should handle multi-turn conversations with history", async () => {
      try {
        const response = await client.chat({
          message: "What is my name?",
          model: "command-r-08-2024",
          chatHistory: [
            { role: "USER", message: "My name is TestUser." },
            { role: "CHATBOT", message: "Hello TestUser! Nice to meet you." },
          ],
        });
        console.log("Multi-turn response:", response.text);
      } catch (error) {
        console.log("Multi-turn errored (span still exported):", (error as Error).message);
      }
    }, 30000);

    it("should handle streaming responses", async () => {
      try {
        const stream = await client.chatStream({
          message: "Count from 1 to 3.",
          model: "command-r-08-2024",
        });


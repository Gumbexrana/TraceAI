/**
 * E2E Tests for @traceai/llamaindex
 *
 * These tests run against LlamaIndex with an OpenAI-compatible backend
 * and export spans to the FI backend via register() from @traceai/fi-core.
 *
 * Required environment variables:
 *   FI_API_KEY     - FI platform API key
 *   FI_SECRET_KEY  - FI platform secret key (if required)
 *   GOOGLE_API_KEY - Google API key for OpenAI-compatible endpoint
 *
 * Run with: FI_API_KEY=... GOOGLE_API_KEY=... pnpm test -- --testPathPattern=e2e
 */
import { register, FITracerProvider, ProjectType } from "@traceai/fi-core";
import { LlamaIndexInstrumentation } from "../instrumentation";

const FI_API_KEY = process.env.FI_API_KEY;
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;

const describeE2E = FI_API_KEY ? describe : describe.skip;

describeE2E("LlamaIndex E2E Tests", () => {
  let provider: FITracerProvider;
  let instrumentation: LlamaIndexInstrumentation;

  beforeAll(async () => {
    provider = register({
      projectName: process.env.FI_PROJECT_NAME || "ts-llamaindex-e2e",
      projectType: ProjectType.OBSERVE,
      batch: false,
    });

    instrumentation = new LlamaIndexInstrumentation();
    instrumentation.setTracerProvider(provider);
    instrumentation.enable();

    const llamaindexModule = await import("llamaindex");
    instrumentation.manuallyInstrument(llamaindexModule as any);
  });

  afterAll(async () => {
    instrumentation.disable();
    await new Promise((resolve) => setTimeout(resolve, 5000));

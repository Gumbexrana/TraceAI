/**
 * E2E tests for HuggingFace instrumentation
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
import { HuggingFaceInstrumentation } from "../instrumentation";

const FI_API_KEY = process.env.FI_API_KEY;

const describeE2E = FI_API_KEY ? describe : describe.skip;

describeE2E("HuggingFace E2E Tests", () => {
  let provider: FITracerProvider;
  let instrumentation: HuggingFaceInstrumentation;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let HfInference: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let client: any;

  beforeAll(async () => {
    provider = register({
      projectName: process.env.FI_PROJECT_NAME || "ts-huggingface-e2e",
      projectType: ProjectType.OBSERVE,
      batch: false,
    });

    instrumentation = new HuggingFaceInstrumentation();
    instrumentation.setTracerProvider(provider);
    instrumentation.enable();

    const hfModule = await import("@huggingface/inference");
    HfInference = hfModule.HfInference;

    instrumentation.manuallyInstrument(hfModule);


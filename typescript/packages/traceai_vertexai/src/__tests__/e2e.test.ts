/**
 * E2E Tests for @traceai/vertexai
 *
 * These tests export spans to the FI backend via register() from @traceai/fi-core.
 * Even error spans (from dummy credentials) appear in the UI.
 *
 * Required environment variables:
 *   FI_API_KEY                    - FI platform API key
 *
 * Run with: FI_API_KEY=... pnpm test -- --testPathPattern=e2e
 */
import { register, FITracerProvider, ProjectType } from "@traceai/fi-core";
import { VertexAIInstrumentation } from "../instrumentation";

const FI_API_KEY = process.env.FI_API_KEY;
const GOOGLE_CLOUD_PROJECT = process.env.GOOGLE_CLOUD_PROJECT;

const describeE2E = FI_API_KEY ? describe : describe.skip;

describeE2E("Vertex AI E2E Tests", () => {
  let provider: FITracerProvider;
  let instrumentation: VertexAIInstrumentation;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let VertexAI: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let vertexAI: any;

  beforeAll(async () => {
    provider = register({
      projectName: process.env.FI_PROJECT_NAME || "ts-vertexai-e2e",
      projectType: ProjectType.OBSERVE,
      batch: false,
    });

    instrumentation = new VertexAIInstrumentation();
    instrumentation.setTracerProvider(provider);
    instrumentation.enable();

    const vertexModule = await import("@google-cloud/vertexai");
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    instrumentation.manuallyInstrument(vertexModule as any);
    VertexAI = vertexModule.VertexAI;
    vertexAI = new VertexAI({
      project: GOOGLE_CLOUD_PROJECT || "dummy-project-for-e2e",
      location: "us-central1",
    });
  });

  afterAll(async () => {
    instrumentation.disable();
    await new Promise((resolve) => setTimeout(resolve, 5000));
    await provider.forceFlush();
    await provider.shutdown();
  }, 15000);


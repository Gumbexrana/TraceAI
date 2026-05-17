/**
 * E2E Tests for @traceai/strands
 *
 * These tests run against the Strands Agents SDK and export spans to the FI backend.
 * Set FI_API_KEY, AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY environment variables to run.
 *
 * Run with: FI_API_KEY=... AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... pnpm test -- --testPathPattern=e2e
 */

import { register, FITracerProvider, ProjectType } from "@traceai/fi-core";
import { StrandsInstrumentation } from "../instrumentation";

const FI_API_KEY = process.env.FI_API_KEY;
const AWS_ACCESS_KEY_ID = process.env.AWS_ACCESS_KEY_ID;
const AWS_SECRET_ACCESS_KEY = process.env.AWS_SECRET_ACCESS_KEY;
const describeE2E = FI_API_KEY ? describe : describe.skip;

describeE2E("Strands E2E Tests", () => {
  let provider: FITracerProvider;
  let instrumentation: StrandsInstrumentation;

  beforeAll(async () => {
    provider = register({
      projectName: process.env.FI_PROJECT_NAME || "ts-strands-e2e",
      projectType: ProjectType.OBSERVE,
      batch: false,
    });

    instrumentation = new StrandsInstrumentation();
    instrumentation.setTracerProvider(provider);
    instrumentation.enable();

    const strandsModule = await import("@strands-agents/sdk");
    instrumentation.manuallyInstrument(strandsModule);
  });

  afterAll(async () => {
    instrumentation.disable();
    await new Promise((resolve) => setTimeout(resolve, 5000));
    await provider.forceFlush();
    await provider.shutdown();
  }, 15000);

  describe("Agent", () => {
    it("should create and run a basic agent", async () => {
      const strandsModule = await import("@strands-agents/sdk");
      const Agent = (strandsModule as any).Agent;

      const agent = new Agent({
        systemPrompt: "You are a helpful assistant. Answer briefly.",
      });

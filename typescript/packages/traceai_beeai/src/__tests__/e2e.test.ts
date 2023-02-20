/**
 * E2E Tests for @traceai/beeai
 *
 * These tests run against the BeeAI framework and export spans to the FI backend.
 * Set FI_API_KEY environment variable to run these tests.
 *
 * Run with: FI_API_KEY=... pnpm test -- --testPathPattern=e2e
 */

import { register, FITracerProvider, ProjectType } from "@traceai/fi-core";
import { BeeAIInstrumentation } from "../instrumentation";

const FI_API_KEY = process.env.FI_API_KEY;
const describeE2E = FI_API_KEY ? describe : describe.skip;

describeE2E("BeeAI E2E Tests", () => {
  let provider: FITracerProvider;
  let instrumentation: BeeAIInstrumentation;

  beforeAll(async () => {
    provider = register({
      projectName: process.env.FI_PROJECT_NAME || "ts-beeai-e2e",
      projectType: ProjectType.OBSERVE,
      batch: false,
    });

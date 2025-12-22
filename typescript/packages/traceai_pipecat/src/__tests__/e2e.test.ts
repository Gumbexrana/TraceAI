/**
 * E2E Tests for @traceai/pipecat
 *
 * These tests run against the Pipecat voice AI pipeline SDK and export spans
 * to the FI backend. Set FI_API_KEY environment variable to run these tests.
 *
 * Run with: FI_API_KEY=... pnpm test -- --testPathPattern=e2e
 */

import { register, FITracerProvider, ProjectType } from "@traceai/fi-core";
import { PipecatInstrumentation } from "../instrumentation";

const FI_API_KEY = process.env.FI_API_KEY;
const describeE2E = FI_API_KEY ? describe : describe.skip;

describeE2E("Pipecat E2E Tests", () => {
  let provider: FITracerProvider;
  let instrumentation: PipecatInstrumentation;

  beforeAll(async () => {
    provider = register({
      projectName: process.env.FI_PROJECT_NAME || "ts-pipecat-e2e",
      projectType: ProjectType.OBSERVE,
      batch: false,

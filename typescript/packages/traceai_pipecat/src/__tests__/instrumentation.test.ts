/**
 * Tests for Pipecat instrumentation.
 */

import { PipecatInstrumentation, isPatched, _resetPatchedStateForTesting } from "../instrumentation";
import { VERSION } from "../version";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";

describe("PipecatInstrumentation", () => {
  const memoryExporter = new InMemorySpanExporter();
  const tracerProvider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(memoryExporter)]
  });
  tracerProvider.register();

  const instrumentation = new PipecatInstrumentation();
  instrumentation.disable();
  instrumentation.setTracerProvider(tracerProvider);

  beforeEach(() => {
    memoryExporter.reset();
    _resetPatchedStateForTesting();
  });

  afterAll(() => {
    instrumentation.disable();
  });

  describe("VERSION", () => {
    it("should export VERSION constant", () => {
      expect(VERSION).toBe("0.1.0");
    });
  });

  describe("PipecatInstrumentation", () => {
    it("should create instrumentation instance", () => {
      const inst = new PipecatInstrumentation();
      expect(inst).toBeInstanceOf(PipecatInstrumentation);
    });

    it("should accept configuration options", () => {
      const inst = new PipecatInstrumentation({
        traceConfig: {
          hideInputs: true,
          hideOutputs: true,
        },
      });
      expect(inst).toBeInstanceOf(PipecatInstrumentation);
    });

    it("should have instrumentationName", () => {
      const inst = new PipecatInstrumentation();
      expect(inst.instrumentationName).toBe("@traceai/fi-instrumentation-pipecat");
    });

    it("should have instrumentationVersion", () => {
      const inst = new PipecatInstrumentation();
      expect(inst.instrumentationVersion).toBe(VERSION);
    });
  });

  describe("isPatched", () => {
    it("should return false when not patched", () => {
      instrumentation.disable();
      expect(isPatched()).toBe(false);
    });
  });

  describe("manuallyInstrument", () => {
    it("should handle null module", () => {
      const inst = new PipecatInstrumentation();
      inst.setTracerProvider(tracerProvider);
      inst.enable();

      // Should not throw
      inst.manuallyInstrument(null);
      expect(true).toBe(true);
    });

    it("should mark module as patched after manual instrumentation", () => {
      const inst = new PipecatInstrumentation();
      inst.setTracerProvider(tracerProvider);
      inst.enable();

      const mockModule = { RTVIClient: {} };
      inst.manuallyInstrument(mockModule);
      expect(mockModule).toHaveProperty("_fiPatched", true);
    });
  });

  describe("enable/disable", () => {
    it("should enable instrumentation", () => {
      const inst = new PipecatInstrumentation();
      inst.setTracerProvider(tracerProvider);
      inst.enable();
      // Should not throw
      expect(true).toBe(true);
    });

    it("should disable instrumentation", () => {
      const inst = new PipecatInstrumentation();
      inst.setTracerProvider(tracerProvider);
      inst.enable();
      inst.disable();
      // Should not throw
      expect(true).toBe(true);
    });
  });
});

describe("Pipecat Client Patching", () => {
  const memoryExporter = new InMemorySpanExporter();
  const tracerProvider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(memoryExporter)]
  });
  tracerProvider.register();

  beforeEach(() => {
    memoryExporter.reset();
    _resetPatchedStateForTesting();
  });

  it("should wrap RTVIClient.connect when available", () => {
    const inst = new PipecatInstrumentation();
    inst.setTracerProvider(tracerProvider);
    inst.enable();

    const originalConnect = jest.fn().mockResolvedValue(undefined);

    const mockModule = {
      RTVIClient: {
        prototype: {
          connect: originalConnect,
        },
      },
    };

    inst.manuallyInstrument(mockModule);
    expect((mockModule as any)._fiPatched).toBe(true);
  });

  it("should wrap RTVIClient.disconnect when available", () => {
    const inst = new PipecatInstrumentation();
    inst.setTracerProvider(tracerProvider);
    inst.enable();


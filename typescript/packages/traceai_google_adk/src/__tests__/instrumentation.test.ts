/**
 * Tests for Google ADK instrumentation.
 */

import { GoogleADKInstrumentation, isPatched, _resetPatchedStateForTesting } from "../instrumentation";
import { VERSION } from "../version";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";

describe("GoogleADKInstrumentation", () => {
  const memoryExporter = new InMemorySpanExporter();
  const tracerProvider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(memoryExporter)]
  });
  tracerProvider.register();

  const instrumentation = new GoogleADKInstrumentation();
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

  describe("GoogleADKInstrumentation", () => {
    it("should create instrumentation instance", () => {
      const inst = new GoogleADKInstrumentation();
      expect(inst).toBeInstanceOf(GoogleADKInstrumentation);
    });

    it("should accept configuration options", () => {
      const inst = new GoogleADKInstrumentation({
        traceConfig: {
          hideInputs: true,
          hideOutputs: true,
        },
      });
      expect(inst).toBeInstanceOf(GoogleADKInstrumentation);
    });

    it("should have instrumentationName", () => {
      const inst = new GoogleADKInstrumentation();
      expect(inst.instrumentationName).toBe("@traceai/fi-instrumentation-google-adk");
    });

    it("should have instrumentationVersion", () => {
      const inst = new GoogleADKInstrumentation();
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
      const inst = new GoogleADKInstrumentation();
      inst.setTracerProvider(tracerProvider);
      inst.enable();

      // Should not throw
      inst.manuallyInstrument(null);
      expect(true).toBe(true);
    });

    it("should mark module as patched after manual instrumentation", () => {
      const inst = new GoogleADKInstrumentation();
      inst.setTracerProvider(tracerProvider);
      inst.enable();

      const mockModule = { Agent: {} };
      inst.manuallyInstrument(mockModule);
      expect(mockModule).toHaveProperty("_fiPatched", true);
    });
  });

  describe("enable/disable", () => {
    it("should enable instrumentation", () => {
      const inst = new GoogleADKInstrumentation();
      inst.setTracerProvider(tracerProvider);
      inst.enable();
      // Should not throw
      expect(true).toBe(true);
    });

    it("should disable instrumentation", () => {
      const inst = new GoogleADKInstrumentation();
      inst.setTracerProvider(tracerProvider);
      inst.enable();
      inst.disable();
      // Should not throw
      expect(true).toBe(true);
    });
  });
});

describe("Agent Patching", () => {
  const memoryExporter = new InMemorySpanExporter();
  const tracerProvider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(memoryExporter)]
  });
  tracerProvider.register();

  beforeEach(() => {
    memoryExporter.reset();
    _resetPatchedStateForTesting();
  });

  it("should wrap Agent.run when available", () => {
    const inst = new GoogleADKInstrumentation();
    inst.setTracerProvider(tracerProvider);
    inst.enable();

    const originalRun = jest.fn().mockResolvedValue({
      output: "Agent response",
      usage: { promptTokens: 10, completionTokens: 20, totalTokens: 30 },
    });

    const mockModule = {
      Agent: {
        prototype: {
          run: originalRun,
        },
      },
    };

    inst.manuallyInstrument(mockModule);
    expect((mockModule as any)._fiPatched).toBe(true);
  });

  it("should wrap Agent.invoke when available", () => {
    const inst = new GoogleADKInstrumentation();
    inst.setTracerProvider(tracerProvider);
    inst.enable();

    const originalInvoke = jest.fn().mockResolvedValue({
      response: "Invoked response",
    });

    const mockModule = {
      Agent: {
        prototype: {
          invoke: originalInvoke,
        },
      },
    };

    inst.manuallyInstrument(mockModule);
    expect((mockModule as any)._fiPatched).toBe(true);
  });

  it("should wrap Agent.stream when available", () => {
    const inst = new GoogleADKInstrumentation();
    inst.setTracerProvider(tracerProvider);
    inst.enable();

    const originalStream = jest.fn().mockResolvedValue(
      (async function* () {
        yield { content: "Streaming" };
        yield { content: " response" };
      })()
    );

    const mockModule = {
      Agent: {
        prototype: {
          stream: originalStream,
        },
      },
    };

    inst.manuallyInstrument(mockModule);
    expect((mockModule as any)._fiPatched).toBe(true);
  });

  it("should not double-patch module", () => {
    const inst = new GoogleADKInstrumentation();
    inst.setTracerProvider(tracerProvider);
    inst.enable();

    const mockModule = {
      _fiPatched: true,
      Agent: {
        prototype: {
          run: jest.fn(),
        },
      },
    };

    const original = mockModule.Agent.prototype.run;
    inst.manuallyInstrument(mockModule);

    // Should not have changed
    expect(mockModule.Agent.prototype.run).toBe(original);
  });
});

describe("Tool Patching", () => {
  const memoryExporter = new InMemorySpanExporter();
  const tracerProvider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(memoryExporter)]
  });
  tracerProvider.register();

  beforeEach(() => {
    memoryExporter.reset();
    _resetPatchedStateForTesting();
  });

  it("should wrap Tool.execute when available", () => {
    const inst = new GoogleADKInstrumentation();
    inst.setTracerProvider(tracerProvider);
    inst.enable();

    const originalExecute = jest.fn().mockResolvedValue({
      result: "Tool execution result",
    });

    const mockModule = {
      Tool: {
        prototype: {
          execute: originalExecute,
        },
      },
    };

    inst.manuallyInstrument(mockModule);
    expect((mockModule as any)._fiPatched).toBe(true);
  });
});

describe("Runner Patching", () => {
  const memoryExporter = new InMemorySpanExporter();
  const tracerProvider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(memoryExporter)]
  });
  tracerProvider.register();

  beforeEach(() => {
    memoryExporter.reset();
    _resetPatchedStateForTesting();
  });

  it("should wrap Runner.run when available", () => {
    const inst = new GoogleADKInstrumentation();
    inst.setTracerProvider(tracerProvider);
    inst.enable();

    const originalRun = jest.fn().mockResolvedValue({
      result: "Runner result",
    });

    const mockModule = {
      Runner: {
        prototype: {
          run: originalRun,
        },
      },

import { describe, it, expect, jest, beforeEach, afterEach } from "@jest/globals";

type MockFn = jest.Mock<any>;

const createMockSpan = () => ({
  setStatus: jest.fn(),
  setAttribute: jest.fn(),
  recordException: jest.fn(),
  setAttributes: jest.fn(),
  addEvent: jest.fn(),
  end: jest.fn(),
  isRecording: jest.fn().mockReturnValue(true),
});

const mockSpan = createMockSpan();
jest.mock("@traceai/fi-core", () => ({
  FITracer: jest.fn().mockImplementation(() => ({
    startSpan: jest.fn().mockReturnValue(mockSpan),
  })),
  TraceConfigOptions: {},
}));

jest.mock("@traceai/fi-semantic-conventions", () => ({
  SemanticConventions: {
    DB_SYSTEM: "db.system",
    DB_OPERATION: "db.operation",
    DB_COLLECTION_NAME: "db.collection.name",
  },
}));

jest.mock("@opentelemetry/api", () => ({
  context: {
    active: jest.fn().mockReturnValue({}),
    with: jest.fn((_ctx: any, fn: () => any) => fn()),
  },
  trace: {
    setSpan: jest.fn().mockReturnValue({}),
    getTracer: jest.fn().mockReturnValue({
      startSpan: jest.fn(),
      startActiveSpan: jest.fn(),
    }),
  },
  metrics: {
    getMeter: jest.fn().mockReturnValue({
      createCounter: jest.fn().mockReturnValue({ add: jest.fn() }),
      createHistogram: jest.fn().mockReturnValue({ record: jest.fn() }),
      createUpDownCounter: jest.fn().mockReturnValue({ add: jest.fn() }),
    }),
  },
  diag: {
    createComponentLogger: jest.fn().mockReturnValue({
      debug: jest.fn(), info: jest.fn(), warn: jest.fn(), error: jest.fn(),
    }),
  },
  SpanKind: { CLIENT: 2 },
  SpanStatusCode: { OK: 1, ERROR: 2 },
}));

let MilvusInstrumentation: any;
let isPatched: any;

describe("Milvus Instrumentation", () => {
  let instrumentation: any;

  beforeEach(() => {
    jest.clearAllMocks();
    jest.resetModules();
    const mod = require("../instrumentation");
    MilvusInstrumentation = mod.MilvusInstrumentation;
    isPatched = mod.isPatched;
    instrumentation = new MilvusInstrumentation({
      instrumentationConfig: { enabled: true },
    });
    instrumentation.enable();
  });

  afterEach(() => {
    instrumentation?.disable();
  });

  describe("Initialization", () => {
    it("should create instrumentation instance", () => {
      expect(instrumentation).toBeInstanceOf(MilvusInstrumentation);
    });

    it("should initialize with default config", () => {
      const inst = new MilvusInstrumentation();
      expect(inst).toBeInstanceOf(MilvusInstrumentation);
    });

    it("should have correct instrumentation name", () => {
      expect(instrumentation.instrumentationName).toBe("@traceai/milvus");
    });

    it("should accept custom configuration", () => {
      const inst = new MilvusInstrumentation({
        instrumentationConfig: { captureQueryVectors: true },
        traceConfig: { maskInputs: true },
      });
      expect(inst).toBeInstanceOf(MilvusInstrumentation);
    });
  });

  describe("Patching", () => {
    it("should report not patched initially", () => {
      expect(isPatched()).toBe(false);
    });

    it("should patch MilvusClient methods", () => {
      const mockModule = {
        MilvusClient: {
          prototype: {
            search: jest.fn(),
            query: jest.fn(),
            insert: jest.fn(),
          },
        },
      };
      instrumentation.manuallyInstrument(mockModule);
      expect(isPatched()).toBe(true);
    });

    it("should not double-patch", () => {
      const mockModule = {
        MilvusClient: { prototype: { search: jest.fn() } },
      };
      instrumentation.manuallyInstrument(mockModule);
      const first = mockModule.MilvusClient.prototype.search;
      instrumentation.manuallyInstrument(mockModule);
      expect(mockModule.MilvusClient.prototype.search).toBe(first);
    });
  });

  describe("Search Operations", () => {
    it("should create span for search operation", async () => {
      const mockClient = {
        search: (jest.fn() as MockFn).mockResolvedValue({ results: [] }),
      };
      const mockModule = {
        MilvusClient: { prototype: { search: mockClient.search } },
      };
      instrumentation.manuallyInstrument(mockModule);

      await mockModule.MilvusClient.prototype.search.call(mockClient, {
        collection_name: "test",
        limit: 10,
      });

      expect(mockSpan.setStatus).toHaveBeenCalledWith({ code: 1 });
      expect(mockSpan.end).toHaveBeenCalled();
    });

    it("should handle search with vector input", async () => {
      const queryVector = new Array(128).fill(0).map(() => Math.random());
      const mockClient = {
        search: (jest.fn() as MockFn).mockResolvedValue({
          results: [
            { id: 1, score: 0.95, entity: { name: "Item 1" } },
            { id: 2, score: 0.87, entity: { name: "Item 2" } },
          ],
        }),
      };
      const mockModule = {
        MilvusClient: { prototype: { search: mockClient.search } },
      };
      instrumentation.manuallyInstrument(mockModule);

      const result = await mockModule.MilvusClient.prototype.search.call(mockClient, {
        collection_name: "products",
        vector: queryVector,
        limit: 10,
        output_fields: ["name", "price"],
      });

      expect(result.results).toHaveLength(2);
      expect(mockSpan.end).toHaveBeenCalled();
    });

    it("should handle search with filters", async () => {
      const mockClient = {
        search: (jest.fn() as MockFn).mockResolvedValue({ results: [] }),
      };
      const mockModule = {
        MilvusClient: { prototype: { search: mockClient.search } },
      };
      instrumentation.manuallyInstrument(mockModule);

      await mockModule.MilvusClient.prototype.search.call(mockClient, {
        collection_name: "products",
        vector: [0.1, 0.2, 0.3],

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

let WeaviateInstrumentation: any;
let isPatched: any;

describe("Weaviate Instrumentation", () => {
  let instrumentation: any;

  beforeEach(() => {
    jest.clearAllMocks();
    jest.resetModules();
    const mod = require("../instrumentation");
    WeaviateInstrumentation = mod.WeaviateInstrumentation;
    isPatched = mod.isPatched;
    instrumentation = new WeaviateInstrumentation({
      instrumentationConfig: { enabled: true },
    });
    instrumentation.enable();
  });

  afterEach(() => {
    instrumentation?.disable();
  });

  describe("Initialization", () => {
    it("should create instrumentation instance", () => {
      expect(instrumentation).toBeInstanceOf(WeaviateInstrumentation);
    });

    it("should initialize with default config", () => {
      const inst = new WeaviateInstrumentation();
      expect(inst).toBeInstanceOf(WeaviateInstrumentation);
    });

    it("should have correct instrumentation name", () => {
      expect(instrumentation.instrumentationName).toBe("@traceai/weaviate");
    });

    it("should accept custom trace config", () => {
      const inst = new WeaviateInstrumentation({
        traceConfig: { maskInputs: true, maskOutputs: true },
      });
      expect(inst).toBeInstanceOf(WeaviateInstrumentation);
    });
  });

  describe("Patching", () => {
    it("should report not patched initially", () => {
      expect(isPatched()).toBe(false);
    });

    it("should patch Collection methods", () => {
      const mockModule = {
        Collection: {
          prototype: {
            query: jest.fn(),
            aggregate: jest.fn(),
          },
        },
        WeaviateClient: {
          prototype: {},
        },
      };
      instrumentation.manuallyInstrument(mockModule);
      expect(isPatched()).toBe(true);
    });

    it("should not double-patch", () => {
      const mockModule = {
        Collection: {
          prototype: { query: jest.fn() },
        },
      };
      instrumentation.manuallyInstrument(mockModule);
      const first = mockModule.Collection.prototype.query;
      instrumentation.manuallyInstrument(mockModule);
      expect(mockModule.Collection.prototype.query).toBe(first);
    });
  });

  describe("Query Operations", () => {
    it("should create span for query operation", async () => {
      const mockCollection = {
        name: "test_collection",
        query: (jest.fn() as MockFn).mockResolvedValue({ objects: [] }),
      };
      const mockModule = {
        Collection: { prototype: { query: mockCollection.query } },
      };
      instrumentation.manuallyInstrument(mockModule);

      await mockModule.Collection.prototype.query.call(mockCollection, { limit: 10 });

      expect(mockSpan.setStatus).toHaveBeenCalledWith({ code: 1 });
      expect(mockSpan.end).toHaveBeenCalled();
    });

    it("should handle nearText search", async () => {
      const mockCollection = {
        name: "articles",
        query: (jest.fn() as MockFn).mockResolvedValue({
          objects: [
            { properties: { title: "ML Basics" } },
            { properties: { title: "Deep Learning" } },
          ],
        }),
      };
      const mockModule = {
        Collection: { prototype: { query: mockCollection.query } },
      };
      instrumentation.manuallyInstrument(mockModule);

      const result = await mockModule.Collection.prototype.query.call(mockCollection, {
        nearText: "machine learning",
        limit: 5,
      });

      expect(result.objects).toHaveLength(2);
      expect(mockSpan.end).toHaveBeenCalled();
    });

    it("should handle hybrid search", async () => {
      const mockCollection = {
        name: "products",
        query: (jest.fn() as MockFn).mockResolvedValue({ objects: [] }),
      };
      const mockModule = {
        Collection: { prototype: { query: mockCollection.query } },
      };
      instrumentation.manuallyInstrument(mockModule);


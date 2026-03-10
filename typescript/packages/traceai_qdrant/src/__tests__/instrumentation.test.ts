import { describe, it, expect, jest, beforeEach, afterEach } from "@jest/globals";

type MockFn = jest.Mock<any>;

// Mock span for testing
const createMockSpan = () => ({
  setStatus: jest.fn(),
  setAttribute: jest.fn(),
  recordException: jest.fn(),
  setAttributes: jest.fn(),
  addEvent: jest.fn(),
  end: jest.fn(),
  isRecording: jest.fn().mockReturnValue(true),
  getSpanContext: jest.fn().mockReturnValue({
    traceId: "mock-trace-id",
    spanId: "mock-span-id",
    traceFlags: 1,
  }),
});

// Mock FITracer
const mockSpan = createMockSpan();
jest.mock("@traceai/fi-core", () => ({
  FITracer: jest.fn().mockImplementation(() => ({
    startSpan: jest.fn().mockReturnValue(mockSpan),
    startActiveSpan: jest.fn((name: any, fn: any) => fn(mockSpan)),
  })),
  TraceConfigOptions: {},
}));

// Mock OpenTelemetry context
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
      debug: jest.fn(),
      info: jest.fn(),
      warn: jest.fn(),
      error: jest.fn(),
      verbose: jest.fn(),
    }),
    debug: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    verbose: jest.fn(),
  },
  SpanKind: {
    CLIENT: 2,
  },
  SpanStatusCode: {
    OK: 1,
    ERROR: 2,
  },
}));

// Import after mocks
let QdrantInstrumentation: any;
let isPatched: any;
let FITracer: any;

describe("Qdrant Instrumentation", () => {
  let instrumentation: any;

  beforeEach(() => {
    jest.clearAllMocks();
    jest.resetModules();

    const instrumentationModule = require("../instrumentation");
    QdrantInstrumentation = instrumentationModule.QdrantInstrumentation;
    isPatched = instrumentationModule.isPatched;
    FITracer = require("@traceai/fi-core").FITracer;

    instrumentation = new QdrantInstrumentation({
      instrumentationConfig: {
        enabled: true,
      },
      traceConfig: {
        maskInputs: false,
        maskOutputs: false,
      },
    });

    instrumentation.enable();
  });

  afterEach(() => {
    if (instrumentation) {
      instrumentation.disable();
    }
  });

  describe("Initialization", () => {
    it("should create instrumentation instance", () => {
      expect(instrumentation).toBeInstanceOf(QdrantInstrumentation);
    });

    it("should initialize with default config", () => {
      const defaultInstrumentation = new QdrantInstrumentation();
      expect(defaultInstrumentation).toBeInstanceOf(QdrantInstrumentation);
    });

    it("should initialize with custom config", () => {
      const customInstrumentation = new QdrantInstrumentation({
        instrumentationConfig: {
          enabled: true,
          captureQueryVectors: true,
          captureResultVectors: false,
        },
        traceConfig: {
          maskInputs: true,
          maskOutputs: true,
        },
      });
      expect(customInstrumentation).toBeInstanceOf(QdrantInstrumentation);
    });

    it("should have correct instrumentation name", () => {
      expect(instrumentation.instrumentationName).toBe("@traceai/qdrant");
    });
  });

  describe("Patching", () => {
    it("should report not patched initially", () => {
      expect(isPatched()).toBe(false);
    });

    it("should patch QdrantClient class methods", () => {
      const mockClient = {
        QdrantClient: {
          prototype: {
            search: jest.fn(),
            query: jest.fn(),
            queryPoints: jest.fn(),
            upsert: jest.fn(),
            delete: jest.fn(),
            retrieve: jest.fn(),
            scroll: jest.fn(),
            count: jest.fn(),
            getCollection: jest.fn(),
          },
        },
      };

      instrumentation.manuallyInstrument(mockClient);

      expect(isPatched()).toBe(true);
      expect(mockClient.QdrantClient.prototype.search).toBeDefined();
      expect(mockClient.QdrantClient.prototype.upsert).toBeDefined();
    });

    it("should not double-patch", () => {
      const mockModule = {
        QdrantClient: {
          prototype: {
            search: jest.fn(),
            upsert: jest.fn(),
          },
        },
      };

      instrumentation.manuallyInstrument(mockModule);
      const firstSearch = mockModule.QdrantClient.prototype.search;

      instrumentation.manuallyInstrument(mockModule);
      const secondSearch = mockModule.QdrantClient.prototype.search;

      expect(firstSearch).toBe(secondSearch);
      expect(isPatched()).toBe(true);
    });
  });

  describe("QdrantClient.search Operation", () => {
    it("should create span with correct attributes for search operation", async () => {
      const mockClient = {
        search: (jest.fn() as MockFn).mockResolvedValue([
          { id: 1, score: 0.95, payload: { text: "result 1" } },
          { id: 2, score: 0.89, payload: { text: "result 2" } },
        ]),
      };

      const mockModule = {
        QdrantClient: {
          prototype: {
            search: mockClient.search,
          },
        },
      };

      instrumentation.manuallyInstrument(mockModule);

      const params = {
        vector: new Array(768).fill(0).map(() => Math.random()),
        limit: 10,
        filter: {
          must: [{ key: "category", match: { value: "documents" } }],
        },
        with_payload: true,
        with_vector: false,
        score_threshold: 0.5,
      };

      await mockModule.QdrantClient.prototype.search.call(
        mockClient,
        "test_collection",
        params
      );

      expect(mockSpan.setStatus).toHaveBeenCalledWith({ code: 1 });
      expect(mockSpan.end).toHaveBeenCalled();
    });
  });

  describe("QdrantClient.upsert Operation", () => {
    it("should create span for upsert operation", async () => {
      const mockClient = {
        upsert: (jest.fn() as MockFn).mockResolvedValue({ status: "completed" }),
      };

      const mockModule = {
        QdrantClient: {
          prototype: {
            upsert: mockClient.upsert,
          },
        },
      };

      instrumentation.manuallyInstrument(mockModule);

      const params = {
        points: [
          { id: 1, vector: new Array(768).fill(0.1), payload: { text: "doc1" } },
          { id: 2, vector: new Array(768).fill(0.2), payload: { text: "doc2" } },
          { id: 3, vector: new Array(768).fill(0.3), payload: { text: "doc3" } },
        ],
      };

      await mockModule.QdrantClient.prototype.upsert.call(
        mockClient,
        "test_collection",
        params
      );

      expect(mockSpan.end).toHaveBeenCalled();
    });
  });

  describe("QdrantClient.delete Operation", () => {
    it("should create span for delete by points operation", async () => {
      const mockClient = {
        delete: (jest.fn() as MockFn).mockResolvedValue({ status: "completed" }),
      };

      const mockModule = {
        QdrantClient: {
          prototype: {
            delete: mockClient.delete,
          },
        },
      };

      instrumentation.manuallyInstrument(mockModule);

      const params = {
        points: [1, 2, 3],
      };

      await mockModule.QdrantClient.prototype.delete.call(
        mockClient,
        "test_collection",
        params
      );

      expect(mockSpan.end).toHaveBeenCalled();
    });

    it("should create span for delete by filter operation", async () => {
      const mockClient = {
        delete: (jest.fn() as MockFn).mockResolvedValue({ status: "completed" }),
      };

      const mockModule = {
        QdrantClient: {
          prototype: {
            delete: mockClient.delete,
          },
        },
      };

      instrumentation.manuallyInstrument(mockModule);

      const params = {
        filter: {
          must: [{ key: "status", match: { value: "obsolete" } }],
        },
      };

      await mockModule.QdrantClient.prototype.delete.call(
        mockClient,
        "test_collection",
        params
      );

      expect(mockSpan.end).toHaveBeenCalled();
    });
  });

  describe("QdrantClient.retrieve Operation", () => {
    it("should create span for retrieve operation", async () => {
      const mockClient = {
        retrieve: (jest.fn() as MockFn).mockResolvedValue([
          { id: 1, payload: { text: "doc1" }, vector: [0.1, 0.2] },
          { id: 2, payload: { text: "doc2" }, vector: [0.3, 0.4] },
        ]),
      };

      const mockModule = {
        QdrantClient: {
          prototype: {
            retrieve: mockClient.retrieve,
          },
        },
      };

      instrumentation.manuallyInstrument(mockModule);

      const params = {
        ids: [1, 2],
        with_payload: true,
        with_vector: true,
      };

      await mockModule.QdrantClient.prototype.retrieve.call(
        mockClient,
        "test_collection",
        params
      );

      expect(mockSpan.end).toHaveBeenCalled();
    });
  });

  describe("QdrantClient.scroll Operation", () => {
    it("should create span for scroll operation", async () => {
      const mockClient = {
        scroll: (jest.fn() as MockFn).mockResolvedValue({
          points: [
            { id: 1, payload: { text: "doc1" } },
            { id: 2, payload: { text: "doc2" } },

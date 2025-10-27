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
    DB_STATEMENT: "db.statement",
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

let PgVectorInstrumentation: any;
let isPatched: any;

describe("PgVector Instrumentation", () => {
  let instrumentation: any;

  beforeEach(() => {
    jest.clearAllMocks();
    jest.resetModules();
    const mod = require("../instrumentation");
    PgVectorInstrumentation = mod.PgVectorInstrumentation;
    isPatched = mod.isPatched;
    instrumentation = new PgVectorInstrumentation({
      instrumentationConfig: { enabled: true },
    });
    instrumentation.enable();
  });

  afterEach(() => {
    instrumentation?.disable();
  });

  describe("Initialization", () => {
    it("should create instrumentation instance", () => {
      expect(instrumentation).toBeInstanceOf(PgVectorInstrumentation);
    });

    it("should initialize with default config", () => {
      const inst = new PgVectorInstrumentation();
      expect(inst).toBeInstanceOf(PgVectorInstrumentation);
    });

    it("should have correct instrumentation name", () => {
      expect(instrumentation.instrumentationName).toBe("@traceai/pgvector");
    });

    it("should accept custom configuration", () => {
      const inst = new PgVectorInstrumentation({
        instrumentationConfig: { captureStatements: true },
        traceConfig: { maskInputs: true },
      });
      expect(inst).toBeInstanceOf(PgVectorInstrumentation);
    });
  });

  describe("Patching", () => {
    it("should report not patched initially", () => {
      expect(isPatched()).toBe(false);
    });

    it("should patch Client and Pool query methods", () => {
      const mockModule = {
        Client: {
          prototype: { query: jest.fn() },
        },
        Pool: {
          prototype: { query: jest.fn() },
        },
      };
      instrumentation.manuallyInstrument(mockModule);
      expect(isPatched()).toBe(true);
    });
  });

  describe("Basic Query Operations", () => {
    it("should create span for SELECT query", async () => {
      const mockClient = {
        query: (jest.fn() as MockFn).mockResolvedValue({ rows: [], rowCount: 0 }),
      };
      const mockModule = {
        Client: { prototype: { query: mockClient.query } },
      };
      instrumentation.manuallyInstrument(mockModule);

      await mockModule.Client.prototype.query.call(mockClient, "SELECT * FROM items");

      expect(mockSpan.setStatus).toHaveBeenCalledWith({ code: 1 });
      expect(mockSpan.end).toHaveBeenCalled();
    });

    it("should handle INSERT query", async () => {
      const mockClient = {
        query: (jest.fn() as MockFn).mockResolvedValue({ rowCount: 1 }),
      };
      const mockModule = {
        Client: { prototype: { query: mockClient.query } },
      };
      instrumentation.manuallyInstrument(mockModule);

      await mockModule.Client.prototype.query.call(
        mockClient,
        "INSERT INTO items (name, embedding) VALUES ($1, $2)",
        ["Test Item", JSON.stringify([0.1, 0.2, 0.3])]
      );

      expect(mockSpan.end).toHaveBeenCalled();
    });

    it("should handle UPDATE query", async () => {
      const mockClient = {
        query: (jest.fn() as MockFn).mockResolvedValue({ rowCount: 5 }),
      };
      const mockModule = {
        Client: { prototype: { query: mockClient.query } },
      };
      instrumentation.manuallyInstrument(mockModule);

      await mockModule.Client.prototype.query.call(
        mockClient,
        "UPDATE items SET embedding = $1 WHERE category = $2",
        [JSON.stringify([0.1, 0.2, 0.3]), "electronics"]
      );

      expect(mockSpan.end).toHaveBeenCalled();
    });
  });

  describe("Vector Search Detection", () => {
    it("should detect L2 distance operator", async () => {
      const mockClient = {
        query: (jest.fn() as MockFn).mockResolvedValue({ rows: [], rowCount: 0 }),
      };
      const mockModule = {
        Client: { prototype: { query: mockClient.query } },
      };
      instrumentation.manuallyInstrument(mockModule);

      await mockModule.Client.prototype.query.call(
        mockClient,
        "SELECT * FROM items ORDER BY embedding <-> $1 LIMIT 10",
        [[0.1, 0.2, 0.3]]
      );

      expect(mockSpan.end).toHaveBeenCalled();
    });

    it("should detect cosine distance operator", async () => {
      const mockClient = {
        query: (jest.fn() as MockFn).mockResolvedValue({ rows: [], rowCount: 0 }),
      };
      const mockModule = {
        Client: { prototype: { query: mockClient.query } },
      };
      instrumentation.manuallyInstrument(mockModule);

      await mockModule.Client.prototype.query.call(
        mockClient,
        "SELECT * FROM items ORDER BY embedding <=> $1 LIMIT 5"
      );

      expect(mockSpan.end).toHaveBeenCalled();
    });

    it("should detect inner product operator", async () => {
      const mockClient = {

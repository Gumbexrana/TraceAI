import { describe, it, expect, jest, beforeEach, afterEach } from '@jest/globals';
import { diag, trace, SpanStatusCode } from '@opentelemetry/api';
import { Resource } from '@opentelemetry/resources';
import {
  register,
  FITracerProvider,
  Transport,
  HTTPSpanExporter,
  GRPCSpanExporter,
  checkCustomEvalConfigExists,
  checkCustomEvalTemplateExists,
  PROJECT_NAME,
  PROJECT_TYPE,
  PROJECT_VERSION_NAME,
  PROJECT_VERSION_ID,
  EVAL_TAGS,
  METADATA,
  SESSION_NAME,
} from '../otel';
import { ProjectType, EvalTag, EvalTagType, EvalSpanKind, EvalName, ModelChoices } from '../fi_types';

// Mock fetch globally
const mockFetch = jest.fn() as jest.MockedFunction<typeof fetch>;
global.fetch = mockFetch;

// Mock grpc
jest.mock('@grpc/grpc-js', () => ({
  credentials: {
    createSsl: jest.fn(),
    createInsecure: jest.fn(),
  },
  Metadata: jest.fn().mockImplementation(() => ({
    set: jest.fn(),
    get: jest.fn(),
    add: jest.fn(),
  })),
}));

// Mock the OTLP HTTP trace exporter
jest.mock('@opentelemetry/exporter-trace-otlp-http', () => ({
  OTLPTraceExporter: jest.fn().mockImplementation(() => ({
    export: jest.fn((_spans: any, callback: any) => callback({ code: 0 })),
    shutdown: jest.fn(() => Promise.resolve()),
    forceFlush: jest.fn(() => Promise.resolve()),
  })),
}));

// Mock the generated client
jest.mock('../generated', () => ({
  ObservationSpanControllerClient: jest.fn().mockImplementation(() => ({
    createOtelSpan: jest.fn(),
  })),
  CreateOtelSpanRequest: jest.fn(),
  CreateOtelSpanResponse: jest.fn(),
}));

describe('FI Core - OTEL Module', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockFetch.mockClear();
  });

  afterEach(() => {
    // Clean up any global tracer provider
    if ((global as any).opentelemetry?.trace?.getTracerProvider) {
      try {
        ((global as any).opentelemetry.trace.getTracerProvider() as any)?.shutdown?.();
      } catch (e) {
        // Ignore cleanup errors
      }
    }
  });

  describe('Resource Attributes Constants', () => {
    it('should have correct resource attribute constants', () => {
      expect(PROJECT_NAME).toBe('project_name');
      expect(PROJECT_TYPE).toBe('project_type');
      expect(PROJECT_VERSION_NAME).toBe('project_version_name');
      expect(PROJECT_VERSION_ID).toBe('project_version_id');
      expect(EVAL_TAGS).toBe('eval_tags');
      expect(METADATA).toBe('metadata');
      expect(SESSION_NAME).toBe('session_name');
    });
  });

  describe('Transport Enum', () => {
    it('should have correct transport values', () => {
      expect(Transport.HTTP).toBe('http');
      expect(Transport.GRPC).toBe('grpc');
    });
  });

  describe('HTTPSpanExporter', () => {
    let exporter: HTTPSpanExporter;
    const mockEndpoint = 'https://test.example.com/api/spans';

    beforeEach(() => {
      exporter = new (HTTPSpanExporter as any)({
        endpoint: mockEndpoint,
        headers: { 'Authorization': 'Bearer test-token' },
        verbose: false,
      });
    });

    it('should create HTTPSpanExporter with correct configuration', () => {
      expect(exporter).toBeDefined();
      expect((exporter as any).endpoint).toBe(mockEndpoint);
      expect((exporter as any).headers).toEqual({
        'Content-Type': 'application/json',
        'Authorization': 'Bearer test-token',
      });
    });

    it('should handle successful export', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        statusText: 'OK',
      } as Response);

      const mockSpan = {
        spanContext: () => ({
          traceId: 'test-trace-id',
          spanId: 'test-span-id',
          traceFlags: 1,
        }),
        name: 'test-span',
        startTime: [1000, 0],
        endTime: [2000, 0],
        attributes: { 'test.attr': 'value' },
        events: [],
        status: { code: SpanStatusCode.OK },
        resource: {
          attributes: {
            [PROJECT_NAME]: 'test-project',
            [PROJECT_TYPE]: ProjectType.OBSERVE,
          },
        },  
      };

      const callback = jest.fn();
      exporter.export([mockSpan as any], callback);

      // Wait for async operation
      await new Promise(resolve => setTimeout(resolve, 0));

      expect(mockFetch).toHaveBeenCalledWith(
        mockEndpoint,
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-token',
          }),
          body: expect.stringContaining('test-span'),
        })
      );
    });

    it('should handle export failure', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
      } as Response);

      const mockSpan = {
        spanContext: () => ({
          traceId: 'test-trace-id',
          spanId: 'test-span-id',
          traceFlags: 1,
        }),
        name: 'test-span',
        startTime: [1000, 0],
        endTime: [2000, 0],
        attributes: {},
        events: [],
        status: { code: SpanStatusCode.OK },
        resource: { attributes: {} },
      };

      const callback = jest.fn();
      exporter.export([mockSpan as any], callback);

      // Wait for async operation
      await new Promise(resolve => setTimeout(resolve, 0));

      expect(callback).toHaveBeenCalledWith({
        code: expect.any(Number), // ExportResultCode.FAILED
      });
    });

    it('should handle network error', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      const mockSpan = {
        spanContext: () => ({
          traceId: 'test-trace-id',
          spanId: 'test-span-id',
          traceFlags: 1,
        }),
        name: 'test-span',
        startTime: [1000, 0],
        endTime: [2000, 0],
        attributes: {},
        events: [],
        status: { code: SpanStatusCode.OK },
        resource: { attributes: {} },
      };

      const callback = jest.fn();
      exporter.export([mockSpan as any], callback);

      // Wait for async operation
      await new Promise(resolve => setTimeout(resolve, 0));

      expect(callback).toHaveBeenCalledWith({
        code: expect.any(Number), // ExportResultCode.FAILED
      });
    });
  });

  describe('FITracerProvider', () => {
    it('should create FITracerProvider with default configuration', () => {
      const provider = new FITracerProvider();
      expect(provider).toBeDefined();
      expect((provider as any).transport).toBe(Transport.HTTP);
    });

    it('should create FITracerProvider with custom configuration', () => {
      const customResource = {
        attributes: {
          [PROJECT_NAME]: 'custom-project',
          [PROJECT_TYPE]: ProjectType.OBSERVE,
        }
      } as unknown as Resource;

      const provider = new FITracerProvider({
        resource: customResource,
        verbose: true,
        transport: Transport.GRPC,
        endpoint: 'https://custom-endpoint.com',
        headers: { 'Custom-Header': 'value' },
      });

      expect(provider).toBeDefined();
      expect((provider as any).transport).toBe(Transport.GRPC);
      expect((provider as any).verbose).toBe(true);
    });
  });

  describe('register function', () => {
    it('should register with default options', () => {
      // Set required environment variable
      const originalEnv = process.env.FI_PROJECT_NAME;
      process.env.FI_PROJECT_NAME = 'test-project';
      
      try {
        const provider = register();
        expect(provider).toBeInstanceOf(FITracerProvider);
      } finally {
        // Restore original environment
        if (originalEnv) {
          process.env.FI_PROJECT_NAME = originalEnv;
        } else {
          delete process.env.FI_PROJECT_NAME;
        }
      }
    });

    it('should register with custom options', () => {
      const provider = register({
        projectName: 'test-project',
        projectType: ProjectType.EXPERIMENT,
        projectVersionName: 'v1.0.0',
        evalTags: [{
          type: EvalTagType.OBSERVATION_SPAN,
          value: EvalSpanKind.LLM,
          eval_name: EvalName.CHUNK_ATTRIBUTION,
          config: {},
          custom_eval_name: "Chunk_Attribution",
          mapping: {
            "context": "raw.input",
            "output": "raw.output"
          },
          model: ModelChoices.TURING_SMALL,
          toDict: () => ({
            type: EvalTagType.OBSERVATION_SPAN,
            value: EvalSpanKind.LLM,
            eval_name: EvalName.CHUNK_ATTRIBUTION,
            config: {},
            custom_eval_name: "Chunk_Attribution",
            mapping: {
              "context": "raw.input",
              "output": "raw.output"
            },
            model: ModelChoices.TURING_SMALL
          })
        } as unknown as EvalTag],
        metadata: { env: 'test' },
        batch: true,
        verbose: true,
        transport: Transport.GRPC,
      });

      expect(provider).toBeInstanceOf(FITracerProvider);
    });

    it('should set global tracer provider when requested', () => {
      const provider = register({
        setGlobalTracerProvider: true,
        projectName: 'global-test',
      });

      expect(provider).toBeInstanceOf(FITracerProvider);
      // Note: Hard to test global tracer provider directly without side effects
    });
  });

  describe('checkCustomEvalConfigExists', () => {
    const mockProjectName = 'test-project';
    const mockEvalTags = [{ name: 'test-tag', value: 'test-value' }];

    it('should return true when config exists', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,

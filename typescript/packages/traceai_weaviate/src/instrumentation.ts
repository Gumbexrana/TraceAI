import {
  InstrumentationBase,
  InstrumentationConfig,
  InstrumentationNodeModuleDefinition,
} from "@opentelemetry/instrumentation";
import {
  context,
  trace,
  SpanKind,
  SpanStatusCode,
  Attributes,
} from "@opentelemetry/api";
import { FITracer, TraceConfigOptions } from "@traceai/fi-core";
import { SemanticConventions } from "@traceai/fi-semantic-conventions";

const VERSION = "0.1.0";

// Weaviate operation types
type WeaviateOperation =
  | "nearVector"
  | "nearText"
  | "hybrid"
  | "bm25"
  | "fetchObjects"
  | "insert"
  | "insertMany"
  | "deleteById"
  | "deleteMany"
  | "aggregate"
  | "query";

// Track patching state
let _isPatched = false;

export function isPatched(): boolean {
  return _isPatched;
}

export interface WeaviateInstrumentationConfig extends InstrumentationConfig {
  captureQueryVectors?: boolean;
  captureResultVectors?: boolean;
  captureDocuments?: boolean;
}

export class WeaviateInstrumentation extends InstrumentationBase {
  private fiTracer!: FITracer;
  private _traceConfig?: TraceConfigOptions;
  private _weaviateConfig: WeaviateInstrumentationConfig;

  constructor({
    instrumentationConfig,
    traceConfig,
  }: {
    instrumentationConfig?: WeaviateInstrumentationConfig;
    traceConfig?: TraceConfigOptions;
  } = {}) {
    super("@traceai/weaviate", VERSION, Object.assign({}, instrumentationConfig));
    this._traceConfig = traceConfig;
    this._weaviateConfig = instrumentationConfig || {};
  }

  public override enable(): void {
    super.enable();
    this.fiTracer = new FITracer({
      tracer: this.tracer,
      traceConfig: this._traceConfig,
    });
  }

  protected init() {
    return new InstrumentationNodeModuleDefinition(
      "weaviate-client",
      [">=3.0.0"],
      this.patch.bind(this),
      this.unpatch.bind(this)
    );
  }

  public manuallyInstrument(weaviateModule: any): void {
    this.patch(weaviateModule);
  }

  private patch(moduleExports: any, moduleVersion?: string) {
    if (_isPatched) {
      return moduleExports;
    }

    if (!this.fiTracer) {
      this.fiTracer = new FITracer({
        tracer: this.tracer,
        traceConfig: this._traceConfig,
      });
    }

    // Patch WeaviateClient if it exists
    if (moduleExports.WeaviateClient?.prototype) {
      this._patchClient(moduleExports.WeaviateClient.prototype);
    }

    // Patch collection query methods if they exist
    if (moduleExports.Collection?.prototype) {
      this._patchCollection(moduleExports.Collection.prototype);
    }

    _isPatched = true;
    return moduleExports;
  }

  private unpatch(moduleExports: any, moduleVersion?: string) {
    _isPatched = false;
  }

  private _patchClient(prototype: any): void {
    // Patch client-level methods if needed
  }

  private _patchCollection(prototype: any): void {
    const instrumentation = this;

    // Patch query method (generic queries)
    if (typeof prototype.query === "function") {
      const original = prototype.query;
      prototype.query = function (this: any, ...args: any[]) {
        return instrumentation._wrapCollectionMethod(
          this,
          original,
          "query",
          args
        );
      };
    }

    // Patch nearVector search
    if (prototype.query?.nearVector) {
      const original = prototype.query.nearVector;
      prototype.query.nearVector = function (...args: any[]) {
        return instrumentation._wrapCollectionMethod(
          this,
          original,
          "nearVector",
          args
        );
      };
    }

    // Patch nearText search
    if (prototype.query?.nearText) {
      const original = prototype.query.nearText;
      prototype.query.nearText = function (...args: any[]) {
        return instrumentation._wrapCollectionMethod(
          this,
          original,
          "nearText",
          args
        );
      };
    }

    // Patch hybrid search
    if (prototype.query?.hybrid) {
      const original = prototype.query.hybrid;
      prototype.query.hybrid = function (...args: any[]) {
        return instrumentation._wrapCollectionMethod(
          this,
          original,
          "hybrid",
          args
        );
      };
    }


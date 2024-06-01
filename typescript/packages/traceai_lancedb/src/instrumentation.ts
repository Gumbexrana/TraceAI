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

type LanceDBOperation =
  | "search"
  | "add"
  | "update"
  | "delete"
  | "createTable"
  | "dropTable"
  | "openTable"
  | "tableNames"
  | "countRows";

let _isPatched = false;

export function isPatched(): boolean {
  return _isPatched;
}

export interface LanceDBInstrumentationConfig extends InstrumentationConfig {
  captureQueryVectors?: boolean;
  captureResultVectors?: boolean;
}

export class LanceDBInstrumentation extends InstrumentationBase {
  private fiTracer!: FITracer;
  private _traceConfig?: TraceConfigOptions;
  private _lanceConfig: LanceDBInstrumentationConfig;

  constructor({
    instrumentationConfig,
    traceConfig,
  }: {
    instrumentationConfig?: LanceDBInstrumentationConfig;
    traceConfig?: TraceConfigOptions;
  } = {}) {
    super("@traceai/lancedb", VERSION, Object.assign({}, instrumentationConfig));
    this._traceConfig = traceConfig;
    this._lanceConfig = instrumentationConfig || {};
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
      "@lancedb/lancedb",
      [">=0.1.0"],
      this.patch.bind(this),
      this.unpatch.bind(this)
    );
  }

  public manuallyInstrument(lanceModule: any): void {
    this.patch(lanceModule);
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

    // Patch Connection class
    if (moduleExports.Connection?.prototype) {
      this._patchConnection(moduleExports.Connection.prototype);
    }

    // Patch Table class
    if (moduleExports.Table?.prototype) {
      this._patchTable(moduleExports.Table.prototype);
    }

    _isPatched = true;
    return moduleExports;
  }

  private unpatch(moduleExports: any, moduleVersion?: string) {
    _isPatched = false;
  }

  private _patchConnection(prototype: any): void {
    const instrumentation = this;
    const methods: LanceDBOperation[] = ["createTable", "dropTable", "openTable", "tableNames"];

    for (const method of methods) {
      if (typeof prototype[method] === "function") {
        const original = prototype[method];
        prototype[method] = function (this: any, ...args: any[]) {
          return instrumentation._wrapMethod(this, original, method, args);
        };
      }
    }
  }

  private _patchTable(prototype: any): void {
    const instrumentation = this;
    const methods: LanceDBOperation[] = ["search", "add", "update", "delete", "countRows"];

    for (const method of methods) {
      if (typeof prototype[method] === "function") {
        const original = prototype[method];
        prototype[method] = function (this: any, ...args: any[]) {
          return instrumentation._wrapMethod(this, original, method, args, this.name);
        };
      }
    }
  }

  private getCommonAttributes(operation: LanceDBOperation, tableName?: string): Attributes {
    const attrs: Attributes = {
      "db.system": "lancedb",
      "db.operation.name": operation,
      [SemanticConventions.FI_SPAN_KIND]: "VECTOR_DB",
    };
    if (tableName) {

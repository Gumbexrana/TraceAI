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

type PgVectorOperation = "query" | "insert" | "update" | "delete" | "vectorSearch";

let _isPatched = false;

export function isPatched(): boolean {
  return _isPatched;
}

export interface PgVectorInstrumentationConfig extends InstrumentationConfig {
  captureQueryVectors?: boolean;
  captureResultVectors?: boolean;
  captureQueries?: boolean;
}

export class PgVectorInstrumentation extends InstrumentationBase {
  private fiTracer!: FITracer;
  private _traceConfig?: TraceConfigOptions;
  private _pgConfig: PgVectorInstrumentationConfig;

  constructor({
    instrumentationConfig,
    traceConfig,
  }: {
    instrumentationConfig?: PgVectorInstrumentationConfig;
    traceConfig?: TraceConfigOptions;
  } = {}) {
    super("@traceai/pgvector", VERSION, Object.assign({}, instrumentationConfig));
    this._traceConfig = traceConfig;
    this._pgConfig = instrumentationConfig || {};
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
      "pg",
      [">=8.0.0"],
      this.patch.bind(this),
      this.unpatch.bind(this)
    );
  }

  public manuallyInstrument(pgModule: any): void {
    this.patch(pgModule);
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

    // Patch Client
    if (moduleExports.Client?.prototype?.query) {
      this._patchQuery(moduleExports.Client.prototype);
    }

    // Patch Pool
    if (moduleExports.Pool?.prototype?.query) {
      this._patchQuery(moduleExports.Pool.prototype);
    }

    _isPatched = true;
    return moduleExports;
  }

  private unpatch(moduleExports: any, moduleVersion?: string) {

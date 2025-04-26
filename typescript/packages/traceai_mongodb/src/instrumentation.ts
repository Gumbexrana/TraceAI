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

type MongoDBVectorOperation =
  | "aggregate"
  | "find"
  | "insertOne"
  | "insertMany"
  | "updateOne"
  | "updateMany"
  | "deleteOne"
  | "deleteMany"
  | "createSearchIndex"
  | "dropSearchIndex";

let _isPatched = false;

export function isPatched(): boolean {
  return _isPatched;
}

export interface MongoDBInstrumentationConfig extends InstrumentationConfig {
  captureQueryVectors?: boolean;
  captureResultVectors?: boolean;
}

export class MongoDBInstrumentation extends InstrumentationBase {
  private fiTracer!: FITracer;
  private _traceConfig?: TraceConfigOptions;
  private _mongoConfig: MongoDBInstrumentationConfig;

  constructor({
    instrumentationConfig,
    traceConfig,
  }: {
    instrumentationConfig?: MongoDBInstrumentationConfig;
    traceConfig?: TraceConfigOptions;
  } = {}) {
    super("@traceai/mongodb", VERSION, Object.assign({}, instrumentationConfig));
    this._traceConfig = traceConfig;
    this._mongoConfig = instrumentationConfig || {};
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
      "mongodb",
      [">=5.0.0"],
      this.patch.bind(this),
      this.unpatch.bind(this)
    );
  }

  public manuallyInstrument(mongoModule: any): void {
    this.patch(mongoModule);
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

    if (moduleExports.Collection?.prototype) {
      this._patchCollection(moduleExports.Collection.prototype);
    }


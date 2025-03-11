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

type MilvusOperation =
  | "search"
  | "query"
  | "insert"
  | "upsert"
  | "delete"
  | "get"
  | "createCollection"
  | "dropCollection"
  | "hasCollection"
  | "describeCollection"
  | "createIndex"
  | "dropIndex";

let _isPatched = false;

export function isPatched(): boolean {
  return _isPatched;
}

export interface MilvusInstrumentationConfig extends InstrumentationConfig {
  captureQueryVectors?: boolean;
  captureResultVectors?: boolean;
}

export class MilvusInstrumentation extends InstrumentationBase {
  private fiTracer!: FITracer;
  private _traceConfig?: TraceConfigOptions;
  private _milvusConfig: MilvusInstrumentationConfig;

  constructor({
    instrumentationConfig,
    traceConfig,
  }: {
    instrumentationConfig?: MilvusInstrumentationConfig;
    traceConfig?: TraceConfigOptions;
  } = {}) {
    super("@traceai/milvus", VERSION, Object.assign({}, instrumentationConfig));
    this._traceConfig = traceConfig;
    this._milvusConfig = instrumentationConfig || {};
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
      "@zilliz/milvus2-sdk-node",
      [">=2.0.0"],
      this.patch.bind(this),
      this.unpatch.bind(this)
    );
  }

  public manuallyInstrument(milvusModule: any): void {
    this.patch(milvusModule);

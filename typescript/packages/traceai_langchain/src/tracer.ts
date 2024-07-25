import { BaseTracer, Run } from "@langchain/core/tracers/base";
import {
  SpanKind,
  Span,
  context,
  trace,
  SpanStatusCode,
} from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import { SemanticConventions } from "@traceai/fi-semantic-conventions";
import {
  safelyFlattenAttributes,
  safelyFormatFunctionCalls,
  safelyFormatIO,
  safelyFormatInputMessages,
  safelyFormatLLMParams,
  safelyFormatMetadata,
  safelyFormatOutputMessages,
  safelyFormatPromptTemplate,
  safelyFormatRetrievalDocuments,
  safelyFormatSessionId,
  safelyFormatTokenCounts,
  safelyFormatToolCalls,
  safelyGetFISpanKindFromRunType,
} from "./utils";
import { FITracer } from "@traceai/fi-core";

type RunWithSpan = {
  run: Run;
  span: Span;
};

export class LangChainTracer extends BaseTracer {
  private tracer: FITracer;
  private runs: Record<string, RunWithSpan | undefined> = {};
  constructor(tracer: FITracer) {
    super();
    this.tracer = tracer;
  }
  name: string = "FILangChainTracer";
  protected persistRun(_run: Run): Promise<void> {
    return Promise.resolve();
  }

  /**
   * Called when a new run is created on v0.1.0 of langchain see {@link BaseTracer}
   * @param run the langchain {@link Run} object
   *
   * This method is only available on langchain ^0.1.0 BaseTracer and has been replaced in 0.2 by onRunCreate
   * we support both 0.1 and 0.2 so we need to check if the method exists on the super class before calling it
   */
  protected async _startTrace(run: Run) {
    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-ignore
    if (typeof super._startTrace === "function") {
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      // @ts-ignore
      await super._startTrace(run);
    }
    await this.startTracing(run);
  }

  /**
   * Called when a new run is created on v0.2.0 of langchain see {@link BaseTracer}

import type * as llamaindex from "llamaindex";

import {
  Tracer,
  Span,
  Context,
  SpanStatusCode,
  trace,
  context,
  DiagLogger,
} from "@opentelemetry/api";
import { safeExecuteInTheMiddle } from "@opentelemetry/instrumentation";

import { SemanticConventions, FISpanKind } from "@traceai/fi-semantic-conventions";
import { safelyJSONStringify } from "@traceai/fi-core";

import { LlamaIndexInstrumentationConfig } from "./types";
import { shouldSendPrompts, llmGeneratorWrapper, generatorWrapper } from "./utils";

type LLM = llamaindex.LLM;

type ResponseType = llamaindex.ChatResponse | llamaindex.CompletionResponse;
type AsyncResponseType =
  | AsyncIterable<llamaindex.ChatResponseChunk>
  | AsyncIterable<llamaindex.CompletionResponse>;

function handleResponse<T extends ResponseType>(
    result: T,
    span: Span,
    metadata: llamaindex.LLMMetadata,
    config: LlamaIndexInstrumentationConfig,
    diag: DiagLogger,
  ): T {
    span.setAttribute(SemanticConventions.LLM_MODEL_NAME, metadata.model);

    if (!shouldSendPrompts()) {
      span.setStatus({ code: SpanStatusCode.OK });
      span.end();
      return result;
    }

    try {
      if ((result as llamaindex.ChatResponse).message) {
        const chatResult = result as llamaindex.ChatResponse;
        const outputContent = typeof chatResult.message.content === "string"
          ? chatResult.message.content
          : chatResult.message.content[0]?.type === "text"
            ? (chatResult.message.content[0] as llamaindex.MessageContentTextDetail).text
            : undefined;
        span.setAttribute(
          SemanticConventions.LLM_OUTPUT_MESSAGES,
          safelyJSONStringify([{ role: chatResult.message.role, content: outputContent }]) ?? "[]",
        );
        span.setStatus({ code: SpanStatusCode.OK });
      }
    } catch (e) {
      diag.warn(e as any);
      config.exceptionLogger?.(e as Error);
    }

    span.end();

    return result;
  }

function handleStreamingResponse<T extends AsyncResponseType>(
    result: T,
    span: Span,
    execContext: Context,
    metadata: llamaindex.LLMMetadata,
    config: LlamaIndexInstrumentationConfig,
  ): T {
    span.setAttribute(SemanticConventions.LLM_MODEL_NAME, metadata.model);
    if (!shouldSendPrompts()) {
      span.setStatus({ code: SpanStatusCode.OK });
      span.end();
      return result;
    }

    return llmGeneratorWrapper(result, execContext, (message) => {
      span.setAttribute(
        SemanticConventions.LLM_OUTPUT_MESSAGES,
        safelyJSONStringify([{ role: "assistant", content: message }]) ?? "[]",
      );
      span.setStatus({ code: SpanStatusCode.OK });
      span.end();
    }) as any;
  }

export function chatWrapper({ className }: { className: string },
    config: LlamaIndexInstrumentationConfig,
    diag: DiagLogger,
    tracer: () => Tracer,
) {
    return (original: LLM["chat"]) => {
      return function method(this: LLM, ...args: Parameters<LLM["chat"]>) {
        const params = args[0] as any;
        const messages = params?.messages;
        const streaming = params?.stream;

        const span = tracer()
          .startSpan(`llamaindex.${className}.chat`);

        span.setAttribute(SemanticConventions.FI_SPAN_KIND, FISpanKind.LLM);

        try {
          span.setAttribute(SemanticConventions.LLM_PROVIDER, className);

          span.setAttribute(
            SemanticConventions.LLM_MODEL_NAME,

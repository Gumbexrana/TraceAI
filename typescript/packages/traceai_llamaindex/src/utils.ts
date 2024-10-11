import type * as llamaindex from "llamaindex";
import { context, Context } from "@opentelemetry/api";

import {
  BaseEmbedding,
  BaseSynthesizer,
  LLM,
  BaseRetriever,
} from "llamaindex";

export const shouldSendPrompts = () => {
  return true;
};

// Adopted from https://github.com/open-telemetry/opentelemetry-js/issues/2951#issuecomment-1214587378
export function bindAsyncGenerator<T = unknown, TReturn = any, TNext = unknown>(
  ctx: Context,
  generator: AsyncGenerator<T, TReturn, TNext>,
): AsyncGenerator<T, TReturn, TNext> {
  return {
    next: context.bind(ctx, generator.next.bind(generator)),
    return: context.bind(ctx, generator.return.bind(generator)),
    throw: context.bind(ctx, generator.throw.bind(generator)),

    [Symbol.asyncIterator]() {
      return bindAsyncGenerator(ctx, generator[Symbol.asyncIterator]());
    },

    [Symbol.asyncDispose]() {
      return Promise.resolve();
    },
  };
}

export async function* generatorWrapper(
  streamingResult: AsyncGenerator,
  ctx: Context,
  fn: () => void,
) {
  for await (const chunk of bindAsyncGenerator(ctx, streamingResult)) {
    yield chunk;
  }
  fn();
}

export async function* llmGeneratorWrapper(
  streamingResult:
    | AsyncIterable<llamaindex.ChatResponseChunk>
    | AsyncIterable<llamaindex.CompletionResponse>,
  ctx: Context,
  fn: (message: string) => void,
) {
  let message = "";

  for await (const messageChunk of bindAsyncGenerator(
    ctx,
    streamingResult as AsyncGenerator,
  )) {
    if ((messageChunk as llamaindex.ChatResponseChunk).delta) {
      message += (messageChunk as llamaindex.ChatResponseChunk).delta;
    }
    if ((messageChunk as llamaindex.CompletionResponse).text) {

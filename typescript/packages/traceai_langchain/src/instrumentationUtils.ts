import type * as CallbackManagerModuleV02 from "@langchain/core/callbacks/manager";
import { LangChainTracer } from "./tracer";
import { FITracer } from "@traceai/fi-core";

/**
 * Adds the {@link LangChainTracer} to the callback handlers if it is not already present
 * @param tracer the {@link tracer} to pass into the {@link LangChainTracer} when added to handlers
 * @param handlers the LangChain callback handlers which may be an array of handlers or a CallbackManager
 * @returns the callback handlers with the {@link LangChainTracer} added
 *
 * If the handlers are an array, we add the tracer to the array if it is not already present
 */
export function addTracerToHandlers(
  tracer: FITracer,
  handlers?: CallbackManagerModuleV02.Callbacks,
): CallbackManagerModuleV02.Callbacks;
export function addTracerToHandlers(
  tracer: FITracer,

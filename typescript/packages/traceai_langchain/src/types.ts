import { SemanticConventions } from "@traceai/fi-semantic-conventions";

type LLMMessageToolCall = {
  [SemanticConventions.TOOL_CALL_FUNCTION_NAME]?: string;
  [SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON]?: string;
};

export type LLMMessageToolCalls = {
  [SemanticConventions.MESSAGE_TOOL_CALLS]?: LLMMessageToolCall[];
};

export type LLMMessageFunctionCall = {
  [SemanticConventions.MESSAGE_FUNCTION_CALL_NAME]?: string;
  [SemanticConventions.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON]?: string;
};

export type LLMMessage = LLMMessageToolCalls &
  LLMMessageFunctionCall & {
    [SemanticConventions.MESSAGE_ROLE]?: string;
    [SemanticConventions.MESSAGE_CONTENT]?: string;
  };

export type LLMMessagesAttributes =
  | {

import { SemanticConventions } from "@traceai/fi-semantic-conventions";
import { Attributes } from "@opentelemetry/api";
import {
  ResponseCreateParamsBase,
  ResponseInputItem,
  ResponseOutputItem,
  ResponseStreamEvent,
  Response as ResponseType,
} from "openai/resources/responses/responses";
import { Stream } from "openai/streaming";
import { safelyJSONStringify } from "@traceai/fi-core";

/**
 * Serialize a non-role-based response item to a plain object
 */
function serializeResponseItem(
  item: Exclude<ResponseInputItem | ResponseOutputItem, { role: string }>,
): Record<string, unknown> {
  switch (item.type) {
    case "function_call":
      return {
        role: "assistant",
        tool_calls: [{ id: item.call_id, function: { name: item.name, arguments: item.arguments } }],
      };
    case "function_call_output":
      return { role: "tool", tool_call_id: item.call_id, content: item.output };
    case "reasoning":
      return {
        role: "assistant",
        content: item.summary.map((s) =>
          s.type === "summary_text" ? { type: "summary_text", text: s.text } : { type: s.type },
        ),
      };
    case "item_reference":
      return { type: "item_reference" };
    case "file_search_call":
      if (!item.results) {
        return {
          role: "assistant",
          tool_calls: [{ id: item.id, function: { name: item.type, arguments: JSON.stringify(item.queries) } }],
        };
      }
      return { role: "tool", tool_call_id: item.id, content: JSON.stringify(item.results) };
    case "computer_call":
      return {
        role: "assistant",
        tool_calls: [{ id: item.id, function: { name: item.type, arguments: JSON.stringify(item.action) } }],
      };
    case "computer_call_output":
      return { role: "tool", tool_call_id: item.call_id, content: JSON.stringify(item.output) };
    case "web_search_call":
      return {
        role: "assistant",
        tool_calls: [{ id: item.id, function: { name: item.type } }],
      };
  }
  return {};
}


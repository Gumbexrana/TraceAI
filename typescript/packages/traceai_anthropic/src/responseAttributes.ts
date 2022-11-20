import { SemanticConventions, LLMSystem, LLMProvider, MimeType } from "@traceai/fi-semantic-conventions";
import { Attributes, AttributeValue } from "@opentelemetry/api";
import Anthropic from "@anthropic-ai/sdk";
import { safelyJSONStringify } from "@traceai/fi-core";

/**
 * Extracts input messages as a JSON blob from Anthropic's MessageParam array and system prompt.
 */
export function getAnthropicInputMessagesAttributes(
  params: Anthropic.MessageCreateParams,
): Attributes {
  const messages: Record<string, unknown>[] = [];

  if (params.system) {
    messages.push({
      role: "system",
      content: typeof params.system === 'string' ? params.system : params.system,
    });
  }

  params.messages.forEach((message: Anthropic.MessageParam) => {
    const obj: Record<string, unknown> = { role: message.role };

    if (typeof message.content === "string") {
      obj.content = message.content;
    } else if (Array.isArray(message.content)) {
      obj.content = message.content.map((contentBlock: any) => {
        if (contentBlock.type === "text") {
          return { type: "text", text: contentBlock.text };
        } else if (contentBlock.type === "image") {
          return { type: "image", source: contentBlock.source };
        } else if (contentBlock.type === "tool_use") {
          return contentBlock;
        } else if (contentBlock.type === "tool_result") {
          return contentBlock;
        }
        return { type: contentBlock.type };
      });
    }
    messages.push(obj);
  });

  return {
    [SemanticConventions.LLM_INPUT_MESSAGES]: safelyJSONStringify(messages) ?? "[]",
  };
}

/**
 * Extracts output messages as a JSON blob from an Anthropic Message object.
 */
export function getAnthropicOutputMessagesAttributes(
  response: Anthropic.Message,
): Attributes {
  const outputMessage: Record<string, unknown> = { role: response.role };
  let lastTextContent = "";
  const toolCalls: Record<string, unknown>[] = [];

  if (response.content && Array.isArray(response.content)) {
    response.content.forEach((contentBlock: Anthropic.ContentBlock) => {
      if (contentBlock.type === "text") {
        lastTextContent = (contentBlock as Anthropic.TextBlock).text;
      } else if (contentBlock.type === "tool_use") {
        const toolUseBlock = contentBlock as Anthropic.ToolUseBlock;
        toolCalls.push({
          function: { name: toolUseBlock.name, arguments: safelyJSONStringify(toolUseBlock.input) ?? "{}" },
        });
      }
    });
  }

  if (lastTextContent) {
    outputMessage.content = lastTextContent;
  }
  if (toolCalls.length > 0) {
    outputMessage.tool_calls = toolCalls;
  }

  return {
    [SemanticConventions.LLM_OUTPUT_MESSAGES]: safelyJSONStringify([outputMessage]) ?? "[]",
  };
}

/**
 * Extracts usage attributes from an Anthropic API response.
 */
export function getAnthropicUsageAttributes(
  response: Anthropic.Message | { usage?: Anthropic.Usage },
): Attributes {
  const usage = response.usage;
  if (usage) {

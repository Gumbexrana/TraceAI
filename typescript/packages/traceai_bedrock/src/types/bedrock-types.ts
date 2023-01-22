/**
 * TypeScript type definitions for AWS Bedrock Runtime API structures
 *
 * This file imports official AWS SDK types wherever possible and only defines
 * custom types for structures not exposed by the SDK.
 */

// Import official AWS SDK types
import {
    // Core content and message types from SDK
    ContentBlock,
    Message,
    SystemContentBlock,
    ConversationRole,
  
    // Tool-related types
    Tool,
    ToolUseBlock,
    ToolResultBlock,
    ToolChoice,
    ToolInputSchema,
  
    // Configuration types
    InferenceConfiguration,
  
    // Streaming types
    ContentBlockStart,
    ContentBlockDelta,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
  } from "@aws-sdk/client-bedrock-runtime";
  import { isObjectWithStringKeys } from "@traceai/fi-core";
  
  // Re-export AWS SDK types for convenience
  export {
    ContentBlock,
    Message,
    SystemContentBlock,
    ConversationRole,
    Tool,
    ToolUseBlock,
    ToolResultBlock,
    ToolChoice,
    ToolInputSchema,
    InferenceConfiguration,
    ContentBlockStart,
    ContentBlockDelta,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
  };
  
  // Custom types that extend or aren't available in the SDK
  
  // Flexible type for multi-provider support with runtime validation
  export type InvokeModelRequestBody = Record<string, unknown>;
  export type InvokeModelResponseBody = Record<string, unknown>;
  export type InvocationParameters = Record<string, unknown>;
  
  // Extended conversation role type to support additional roles like "tool" and "system" for Mistral
  export type ExtendedConversationRole = ConversationRole | "tool" | "system";
  
  // Legacy message format for InvokeModel (different from Converse Message)
  export interface BedrockMessage {
    role: ExtendedConversationRole;
    content: MessageContent;
  }
  
  export type MessageContent =
    | string
    | (TextContent | ImageContent | ToolUseContent | ToolResultContent)[];
  
  // Content types for legacy InvokeModel API
  export interface TextContent {
    type: "text";
    text: string;
  }
  
  export interface ImageSource {
    type: "base64";
    media_type: string;
    data: string;
  }
  
  export interface ImageContent {
    type: "image";
    source: ImageSource;
  }
  
  export interface ToolUseContent {
    type: "tool_use";
    id: string;
    name: string;
    input: Record<string, unknown>;
  }
  
  export interface ToolResultContent {
    type: "tool_result";
    tool_use_id: string;
    content: string;
  }
  
  // Tool definition for legacy InvokeModel API
  export interface ToolDefinition {
    name: string;
    description: string;
    input_schema: {
      type: "object";
      properties: Record<string, unknown>;
      required: string[];
    };
  }
  
  // Usage information
  export interface UsageInfo {
    input_tokens: number;
    output_tokens: number;
    cache_read_input_tokens?: number;
    cache_creation_input_tokens?: number;
  }
  
  /**
   * Comprehensive usage attributes normalized from various provider response formats
   * Includes all token count types that may be present across different model providers
   * All fields are optional - undefined means the token count was not provided by the model
   */
  export interface UsageAttributes {
    /** Input/prompt tokens consumed */
    input_tokens?: number;
    /** Output/completion tokens generated */
    output_tokens?: number;
    /** Total tokens (input + output), if provided by the model */
    total_tokens?: number;
    /** Cache read tokens for prompt caching features */
    cache_read_input_tokens?: number;
    /** Cache write tokens for prompt caching features */
    cache_creation_input_tokens?: number;
  }
  
  // Content blocks for Converse API (map to SDK ContentBlock structure)
  // These are used for type guards since AWS SDK ContentBlock has complex union types
  export type ConverseContentBlock =
    | ConverseTextContent
    | ConverseImageContent
    | ConverseToolUseContent
    | ConverseToolResultContent;
  
  export interface ConverseTextContent {
    text: string;
  }
  
  export interface ConverseImageContent {
    image: {
      format: "png" | "jpeg" | "gif" | "webp";
      source: {
        bytes: Uint8Array;
      };
    };
  }
  
  export interface ConverseToolUseContent {
    toolUse: {
      toolUseId: string;
      name: string;
      input: Record<string, unknown>;
    };
  }
  
  export interface ConverseToolResultContent {
    toolResult: {
      toolUseId: string;
      content: ConverseContentBlock[];
      status?: "success" | "error";
    };
  }
  
  // Type guards for legacy InvokeModel content types
  export function isTextContent(content: unknown): content is TextContent {
    return (
      content !== null &&
      typeof content === "object" &&
      "type" in content &&
      content.type === "text" &&
      "text" in content &&
      typeof content.text === "string"
    );
  }
  
  export function isImageContent(content: unknown): content is ImageContent {
    return (
      content !== null &&
      typeof content === "object" &&
      "type" in content &&
      content.type === "image" &&
      "source" in content &&
      content.source !== null &&
      typeof content.source === "object" &&

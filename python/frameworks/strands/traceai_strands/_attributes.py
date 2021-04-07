"""Semantic attributes for Strands Agents instrumentation.

This module provides helper functions to extract attributes from Strands
Agent objects, tools, and model providers following OpenTelemetry GenAI
semantic conventions.
"""

from typing import Any, Dict, List, Optional


# GenAI Semantic Convention attribute names
class SpanAttributes:
    """OpenTelemetry GenAI semantic convention attributes."""

    # LLM attributes
    GEN_AI_PROVIDER_NAME = "gen_ai.system"
    GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
    LLM_RESPONSE_MODEL = "gen_ai.response.model"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"

    # Agent attributes
    GEN_AI_AGENT_NAME = "agent.name"
    AGENT_TYPE = "agent.type"
    AGENT_DESCRIPTION = "agent.description"

    # Tool attributes
    GEN_AI_TOOL_NAME = "gen_ai.tool.name"
    GEN_AI_TOOL_DESCRIPTION = "gen_ai.tool.description"
    TOOL_PARAMETERS = "gen_ai.tool.parameters"
    TOOL_RESULT = "gen_ai.tool.result"

    # Strands-specific attributes
    STRANDS_MODEL_PROVIDER = "strands.model.provider"
    STRANDS_SYSTEM_PROMPT = "strands.system_prompt"
    STRANDS_TOOL_COUNT = "strands.tool_count"
    STRANDS_MCP_SERVER = "strands.mcp.server"
    STRANDS_SESSION_ID = "strands.session.id"
    STRANDS_USER_ID = "strands.user.id"
    STRANDS_CACHE_READ_TOKENS = "strands.cache.read_tokens"
    STRANDS_CACHE_WRITE_TOKENS = "strands.cache.write_tokens"


# Model provider detection patterns - ordered by specificity
# Note: Check specific providers before bedrock (which is a multi-provider platform)
MODEL_PROVIDER_PATTERNS = [
    # Explicit provider prefixes are checked first (see below)
    # Then check specific model patterns
    ("openai", ["gpt-", "o1-", "o3-", "text-davinci", "text-embedding"]),
    ("anthropic", ["claude"]),
    ("google", ["gemini", "palm", "bison"]),
    ("mistral", ["mistral", "mixtral", "codestral"]),
    ("meta", ["llama"]),
    ("cohere", ["command", "embed-"]),
    ("ollama", ["ollama"]),
    # Bedrock-specific patterns (when not matched by above)
    ("bedrock", ["amazon.", "ai21.", "stability."]),
]


def get_model_provider(model_name: Optional[str]) -> str:
    """Detect the model provider from a model name.

    Args:
        model_name: The model name or identifier.

    Returns:
        The detected provider name or "unknown".
    """
    if not model_name:
        return "unknown"

    model_lower = model_name.lower()

    # Check for explicit provider prefixes (e.g., "ollama/llama3", "azure/gpt-4")
    if "/" in model_name:
        prefix = model_name.split("/")[0].lower()
        if prefix in ["bedrock", "azure", "openai", "anthropic", "google", "mistral", "ollama"]:
            return prefix

    # Check for Bedrock ARN patterns (e.g., "us.anthropic.claude-sonnet-4")
    if model_lower.startswith("us.") or model_lower.startswith("eu."):
        return "bedrock"

    # Check for known model patterns in order of specificity
    for provider, patterns in MODEL_PROVIDER_PATTERNS:
        for pattern in patterns:
            if pattern in model_lower:
                return provider

    return "unknown"


def get_agent_attributes(agent: Any) -> Dict[str, Any]:
    """Extract attributes from a Strands Agent instance.


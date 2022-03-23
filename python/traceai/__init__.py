"""
TraceAI - Simple one-liner SDK initialization for FutureAGI observability.

Example usage:
    import traceai
    traceai.init(project_name="my-app", auto_instrument=["openai", "anthropic"])

    # Now use OpenAI/Anthropic normally - traces are automatically captured
    import openai
    client = openai.OpenAI()
    response = client.chat.completions.create(...)
"""

import importlib
from typing import Any, Dict, List, Optional

from fi_instrumentation import (
    SemanticConvention,
    TracerProvider,
    Transport,
    register,
)
from fi_instrumentation.fi_types import ProjectType

# Registry of available instrumentors
_INSTRUMENTOR_REGISTRY: Dict[str, str] = {
    # LLM provider instrumentors
    "openai": "traceai_openai",
    "anthropic": "traceai_anthropic",
    "litellm": "traceai_litellm",
    "groq": "traceai_groq",
    "mistralai": "traceai_mistralai",
    "mistral": "traceai_mistralai",
    "cohere": "traceai_cohere",
    "bedrock": "traceai_bedrock",
    "vertexai": "traceai_vertexai",
    "vertex": "traceai_vertexai",
    "together": "traceai_together",
    "cerebras": "traceai_cerebras",
    "deepseek": "traceai_deepseek",
    "fireworks": "traceai_fireworks",
    "xai": "traceai_xai",
    "ollama": "traceai_ollama",
    "google_genai": "traceai_google_genai",
    "google-genai": "traceai_google_genai",
    # Framework instrumentors
    "langchain": "traceai_langchain",
    "llamaindex": "traceai_llamaindex",
    "llama_index": "traceai_llamaindex",
    "crewai": "traceai_crewai",
    "crew": "traceai_crewai",
    "autogen": "traceai_autogen",
    "dspy": "traceai_dspy",
    "haystack": "traceai_haystack",
    "instructor": "traceai_instructor",
    "guardrails": "traceai_guardrails",
    "mcp": "traceai_mcp",
}

_tracer_provider: Optional[TracerProvider] = None


def init(
    *,
    project_name: Optional[str] = None,
    project_type: str = "experiment",
    project_version_name: Optional[str] = None,
    auto_instrument: Optional[List[str]] = None,
    semantic_convention: str = "fi",
    transport: str = "http",
    batch: bool = False,
    verbose: bool = True,
    set_global_tracer_provider: bool = True,
) -> TracerProvider:
    """
    Initialize TraceAI with a simple one-liner.

    This function registers the tracer provider and optionally instruments
    specified libraries automatically.

    Args:
        project_name: Name of the project (uses FI_PROJECT_NAME env var if not set)
        project_type: Type of project - "experiment" or "observe"
        project_version_name: Optional version name for the project
        auto_instrument: List of libraries to auto-instrument (e.g., ["openai", "anthropic"])
        semantic_convention: Attribute naming convention - "fi", "otel_genai", "openinference", "openllmetry"
        transport: Transport protocol - "http" or "grpc"
        batch: Use batch span processor instead of simple (better for production)
        verbose: Print configuration details
        set_global_tracer_provider: Set as global OpenTelemetry tracer provider

    Returns:
        TracerProvider: The configured tracer provider


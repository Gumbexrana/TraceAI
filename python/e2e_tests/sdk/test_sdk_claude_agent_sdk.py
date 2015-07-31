"""
E2E Tests for Claude Agent SDK Instrumentation

Tests Claude Agent SDK instrumentation. Requires ANTHROPIC_API_KEY and Claude CLI.
"""

import pytest
import time

from config import config, skip_if_no_anthropic


@pytest.fixture(scope="module")
def setup_claude_agent():
    """Set up Claude Agent SDK with instrumentation."""
    if not config.has_anthropic():
        pytest.skip("ANTHROPIC_API_KEY not set")

    from fi_instrumentation import register
    try:
        from traceai_claude_agent_sdk import ClaudeAgentInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_claude_agent_sdk not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    ClaudeAgentInstrumentor().instrument(tracer_provider=tracer_provider)

    yield


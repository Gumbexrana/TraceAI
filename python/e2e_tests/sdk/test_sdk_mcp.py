"""
E2E Tests for MCP (Model Context Protocol) SDK Instrumentation

Tests MCP instrumentation. Requires an MCP server setup.
"""

import pytest
import os
import time

from config import config


HAS_MCP_SERVER = bool(os.getenv("MCP_SERVER_URL"))

skip_if_no_mcp = pytest.mark.skipif(
    not HAS_MCP_SERVER, reason="MCP_SERVER_URL not set"
)


@pytest.fixture(scope="module")
def setup_mcp():
    """Set up MCP with instrumentation."""
    if not HAS_MCP_SERVER:
        pytest.skip("MCP_SERVER_URL not set")

    from fi_instrumentation import register
    try:
        from traceai_mcp import MCPInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_mcp not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

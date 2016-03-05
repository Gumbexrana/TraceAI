"""
Multi-Agent Distributed Trace Demo
===================================

Demonstrates how traceai-a2a stitches two separate agent processes into
a single end-to-end trace in FutureAGI / any OTel backend.

Architecture:
    [Orchestrator Agent] ──A2A──→ [Specialist Agent]
          │                               │
          │  (A2A_CLIENT span)            │  (A2A_SERVER span — child of CLIENT)
          │                               │  (LLM span — child of SERVER)
          └── same trace_id ─────────────┘

The key insight: without traceai-a2a, these would be TWO separate traces
with no link between them. With traceai-a2a, you get ONE trace spanning
both agents.

Usage:
    python multi_agent_demo.py

Requirements:
    - FI_API_KEY and FI_SECRET_KEY set in environment (for FutureAGI backend)
    - OR any OTel-compatible backend configured

This demo works WITHOUT the real a2a-sdk by using mocks — perfect for
seeing the trace structure locally.
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("a2a_demo")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Set up traceAI (works with or without real FI credentials)
# ─────────────────────────────────────────────────────────────────────────────

def setup_tracing(project_name: str = "multi_agent_demo"):
    """Configure the traceAI tracer provider."""
    try:
        from fi_instrumentation import register
        from fi_instrumentation.fi_types import ProjectType

        api_key = os.environ.get("FI_API_KEY", "demo-key")
        secret_key = os.environ.get("FI_SECRET_KEY", "demo-secret")
        os.environ.setdefault("FI_API_KEY", api_key)
        os.environ.setdefault("FI_SECRET_KEY", secret_key)
        os.environ.setdefault(
            "FI_BASE_URL", os.environ.get("FI_BASE_URL", "https://api.futureagi.com")
        )

        trace_provider = register(
            project_type=ProjectType.OBSERVE,
            project_name=project_name,
            verbose=True,
        )
        logger.info("✅ traceAI registered — project: %s", project_name)
        return trace_provider

    except Exception as exc:
        logger.warning("traceAI registration failed (%s) — using local console exporter", exc)
        return _fallback_provider()


def _fallback_provider():
    """Create a simple console-exporting tracer provider for local demo."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider._demo_exporter = exporter  # stash for later inspection
    return provider


# ─────────────────────────────────────────────────────────────────────────────
# 2. Mock A2A Client (simulates the real a2a-sdk A2AClient)
# ─────────────────────────────────────────────────────────────────────────────

class MockA2ATask:
    """Simulates an A2A SDK Task object returned by send_task()."""
    def __init__(self, task_id: str, state: str = "completed"):
        self.id = task_id
        class _Status:
            def __init__(self, state):
                self.state = state
        self.status = _Status(state)


class MockA2AArtifactEvent:
    """Simulates an SSE artifact event from send_task_streaming()."""
    def __init__(self, text: str):
        class _TextPart:
            type = "text"
            text_content = text
        class _Artifact:
            parts = [_TextPart()]
        self.artifact = _Artifact()


class MockA2AStatusEvent:
    """Simulates an SSE status event."""
    def __init__(self, state: str):
        class _Status:
            def __init__(self, s):
                self.state = s

"""
Tests for traceai-a2a: Agent-to-Agent Protocol instrumentation.

Follows exact same patterns as test_framework_openai.py, test_framework_guardrails.py, etc.
Uses in-memory span export + unittest.mock — no real A2A SDK or FI credentials required.

Test coverage:
  - A2AClientWrapper: send_task span creation and attribute recording
  - A2AClientWrapper: send_task_streaming SSE span recording
  - W3C TraceContext propagation (traceparent header injection)
  - A2ATracingMiddleware: incoming context extraction
  - A2AInstrumentor: lifecycle (instrument / uninstrument)
  - Error handling: exception recording and ERROR status
  - Semantic conventions: all A2A span attributes present and correct
  - Streaming span lifecycle: span.end() called even on early-exit (break)
"""

import asyncio
import sys
import types
import uuid
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures and helpers
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def in_memory_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(in_memory_exporter: InMemorySpanExporter) -> TracerProvider:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(in_memory_exporter))
    return provider


@pytest.fixture()
def tracer(tracer_provider: TracerProvider):
    return tracer_provider.get_tracer("test_a2a")


def get_span_attrs(exporter: InMemorySpanExporter, index: int = 0) -> Dict[str, Any]:
    spans = exporter.get_finished_spans()
    assert len(spans) > index, f"Expected at least {index + 1} span(s), got {len(spans)}"
    return dict(spans[index].attributes or {})


def get_spans(exporter: InMemorySpanExporter) -> list:
    return exporter.get_finished_spans()


# ─────────────────────────────────────────────────────────────────────────────
# Mock A2A SDK objects
# ─────────────────────────────────────────────────────────────────────────────

class _MockStatus:
    def __init__(self, state: str = "completed"):
        self.state = state


class _MockTask:
    def __init__(self, task_id: str = None, state: str = "completed"):
        self.id = task_id or str(uuid.uuid4())
        self.status = _MockStatus(state)


class _MockTextPart:
    type = "text"


class _MockArtifact:
    def __init__(self, part_type: str = "text"):
        part = MagicMock()
        part.type = part_type
        self.parts = [part]


class _MockArtifactEvent:
    def __init__(self, part_type: str = "text"):
        self.artifact = _MockArtifact(part_type)
        self.status = None


class _MockStatusEvent:
    def __init__(self, state: str = "completed"):
        self.artifact = None
        self.status = _MockStatus(state)


def _make_mock_a2a_client_class(
    task_id: str = None,
    state: str = "completed",
    streaming_events: Optional[list] = None,
    raise_on_call: Optional[Exception] = None,
):
    """Factory: creates a mock A2AClient class with configurable behavior."""
    task_id = task_id or str(uuid.uuid4())

    class MockA2AClient:
        def __init__(self, url: str = "http://agent.example.com", **kwargs):
            self.url = url
            self._captured_headers: Dict[str, str] = {}

        async def send_task(self, payload: Any = None, **kwargs):
            self._captured_headers = dict(kwargs.get("headers") or {})
            if raise_on_call:
                raise raise_on_call
            return _MockTask(task_id=task_id, state=state)

        async def send_task_streaming(self, payload: Any = None, **kwargs):
            self._captured_headers = dict(kwargs.get("headers") or {})
            if raise_on_call:
                raise raise_on_call
            events = streaming_events or [
                _MockArtifactEvent("text"),
                _MockStatusEvent("completed"),
            ]
            for event in events:
                yield event

    return MockA2AClient


def _inject_mock_a2a_module(client_class):
    """Inject a mock a2a module so A2AInstrumentor can find A2AClient."""
    mock_a2a = types.ModuleType("a2a")
    mock_client_mod = types.ModuleType("a2a.client")
    mock_client_mod.A2AClient = client_class
    mock_a2a.client = mock_client_mod
    mock_a2a.A2AClient = client_class
    sys.modules["a2a"] = mock_a2a
    sys.modules["a2a.client"] = mock_client_mod
    return mock_a2a


def _remove_mock_a2a_module():
    sys.modules.pop("a2a", None)
    sys.modules.pop("a2a.client", None)


# ─────────────────────────────────────────────────────────────────────────────
# Imports — traceai_a2a lives one level up from this tests/ directory
# ─────────────────────────────────────────────────────────────────────────────

import os
_A2A_FRAMEWORK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _A2A_FRAMEWORK_DIR not in sys.path:
    sys.path.insert(0, _A2A_FRAMEWORK_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: send_task creates an A2A_CLIENT span
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a2a_send_task_creates_span(in_memory_exporter, tracer):
    from traceai_a2a._a2a_client import A2AClientWrapper

    MockClient = _make_mock_a2a_client_class()
    instance = MockClient(url="http://agent.example.com:8080")
    wrapper = A2AClientWrapper(tracer=tracer)

    async def fake_send_task(payload=None, **kwargs):
        return await instance.send_task(payload, **kwargs)

    fake_send_task._a2a_streaming = False
    await wrapper.__call_async__(
        wrapped=fake_send_task,
        instance=instance,
        args=({"message": {"role": "user", "parts": []}},),
        kwargs={},
    )

    spans = get_spans(in_memory_exporter)
    assert len(spans) == 1
    assert "A2AClient" in spans[0].name or "send_task" in spans[0].name


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: send_task records task ID as span attribute
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a2a_send_task_records_task_id(in_memory_exporter, tracer):
    from traceai_a2a._a2a_client import A2AClientWrapper
    from traceai_a2a._semantic_conventions import A2A_TASK_ID

    expected_id = str(uuid.uuid4())
    MockClient = _make_mock_a2a_client_class(task_id=expected_id)
    instance = MockClient(url="http://agent.example.com:8080")
    wrapper = A2AClientWrapper(tracer=tracer)

    async def fake_send_task(payload=None, **kwargs):
        return await instance.send_task(payload, **kwargs)

    fake_send_task._a2a_streaming = False
    await wrapper.__call_async__(
        wrapped=fake_send_task,
        instance=instance,
        args=({"message": {"role": "user", "parts": []}},),
        kwargs={},
    )

    attrs = get_span_attrs(in_memory_exporter)
    assert attrs.get(A2A_TASK_ID) == expected_id


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: send_task records agent URL as span attribute
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a2a_send_task_records_agent_url(in_memory_exporter, tracer):
    from traceai_a2a._a2a_client import A2AClientWrapper
    from traceai_a2a._semantic_conventions import A2A_AGENT_URL

    expected_url = "http://specialist-agent.internal:9000"
    MockClient = _make_mock_a2a_client_class()
    instance = MockClient(url=expected_url)
    wrapper = A2AClientWrapper(tracer=tracer)

    async def fake_send_task(payload=None, **kwargs):
        return await instance.send_task(payload, **kwargs)

    fake_send_task._a2a_streaming = False
    await wrapper.__call_async__(
        wrapped=fake_send_task,
        instance=instance,
        args=({"message": {"role": "user", "parts": []}},),
        kwargs={},
    )

    attrs = get_span_attrs(in_memory_exporter)
    assert attrs.get(A2A_AGENT_URL) == expected_url


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: W3C traceparent header is injected into outbound call (KEY TEST)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a2a_send_task_injects_traceparent_header(in_memory_exporter, tracer_provider):
    """
    This is the core test of the feature:
    When A2AClient.send_task is called inside an active span,
    the outbound HTTP call should carry a 'traceparent' header
    that encodes the parent trace/span IDs.
    """
    from traceai_a2a._a2a_client import A2AClientWrapper
    from traceai_a2a._semantic_conventions import A2A_PROPAGATED_TRACE_ID

    captured_headers: Dict[str, str] = {}
    tracer = tracer_provider.get_tracer("test_propagation")

    MockClient = _make_mock_a2a_client_class()
    instance = MockClient(url="http://remote-agent:8080")
    wrapper = A2AClientWrapper(tracer=tracer)

    async def fake_send_task(payload=None, **kwargs):
        # Capture whatever headers were injected by the wrapper

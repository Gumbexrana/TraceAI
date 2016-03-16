"""
A2A Protocol client-side instrumentation.

Wraps A2AClient.send_task() and A2AClient.send_task_streaming() to:
1. Start an A2A_CLIENT span
2. Inject W3C TraceContext (traceparent/tracestate) into outbound HTTP headers
3. Record task/message/artifact attributes on the span
4. Propagate span context so the remote agent's trace is linked

This is the key mechanism that "stitches" distributed multi-agent traces
into a single unified view in your observability backend.

Span lifecycle for streaming calls
-----------------------------------
The reviewer correctly noted that handing span ownership to a bare generator
is unsafe: if the caller does ``for ev in stream: if done: break`` the
generator's ``finally`` block never fires and the span leaks.

Fix: streaming results are wrapped in ``_StreamingSpanWrapper`` /
``_AsyncStreamingSpanWrapper`` — real objects that implement
``__iter__``/``close()`` (and ``__aiter__``/``aclose()``).  The span is
ended inside ``close()``/``aclose()``, and a ``__del__`` guard provides a
last-resort finalizer in case the caller drops the reference without
iterating.
"""

import logging
from contextlib import contextmanager
from typing import Any, AsyncIterator, Dict, Iterator, Mapping, Optional, Tuple

from opentelemetry import context as context_api
from opentelemetry import propagate
from opentelemetry import trace as trace_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import INVALID_SPAN, Span, Status, StatusCode, Tracer
from opentelemetry.util.types import AttributeValue

from traceai_a2a._attributes import (
    get_agent_card_attributes,
    get_artifact_type,
    get_send_task_payload_attributes,
    get_task_attributes,
)
from traceai_a2a._semantic_conventions import (
    A2A_AGENT_URL,
    A2A_ARTIFACT_TYPE,
    A2A_PROPAGATED_TRACE_ID,
    A2A_SPAN_KIND_CLIENT,
    A2A_TASK_ID,
    A2A_TASK_STATE,
    TASK_STATE_COMPLETED,
    TASK_STATE_FAILED,
)

try:
    from fi_instrumentation import get_attributes_from_context
    from fi_instrumentation.fi_types import SpanAttributes
    _FI_AVAILABLE = True
except ImportError:
    _FI_AVAILABLE = False

logger = logging.getLogger(__name__)


class _A2ACarrier(dict):
    """
    A simple dict-based carrier for OTel context propagation.
    Passed to propagate.inject() then merged into the A2A HTTP call headers.
    """
    pass


def _inject_trace_context() -> Tuple[Dict[str, str], Optional[str]]:
    """
    Inject W3C TraceContext into a carrier dict.
    Returns (headers_dict, trace_id_hex_string).

    The headers dict should be merged into the A2A client's HTTP request headers.
    The trace_id is recorded as a span attribute (gen_ai.a2a.propagated_trace_id)
    so you can easily look up the stitched trace by ID.
    """
    carrier = _A2ACarrier()
    propagate.inject(carrier)

    trace_id: Optional[str] = None
    traceparent = carrier.get("traceparent", "")
    if traceparent:
        # traceparent format: 00-<trace-id>-<span-id>-<flags>
        parts = traceparent.split("-")
        if len(parts) == 4:
            trace_id = parts[1]

    return dict(carrier), trace_id


@contextmanager
def _start_a2a_client_span(
    tracer: Tracer,
    agent_url: str,
    span_name: str = "A2AClient.send_task",
    extra_attributes: Optional[Mapping[str, AttributeValue]] = None,
) -> Iterator[Span]:
    """
    Context manager that starts an A2A_CLIENT span.
    Sets the agent URL and span kind attributes immediately on start.
    """
    attributes: Dict[str, AttributeValue] = {
        SpanAttributes.GEN_AI_SPAN_KIND: A2A_SPAN_KIND_CLIENT,
        A2A_AGENT_URL: agent_url,
    }

    if extra_attributes:
        attributes.update(extra_attributes)

    # Inherit any context attributes from the current FI context
    if _FI_AVAILABLE:
        try:
            ctx_attrs = dict(get_attributes_from_context())
            attributes.update(ctx_attrs)
        except Exception:
            pass

    try:
        span = tracer.start_span(name=span_name, attributes=attributes)
    except Exception:
        logger.exception("Failed to start A2A client span")
        span = INVALID_SPAN

    with trace_api.use_span(
        span,
        end_on_exit=False,
        record_exception=False,
        set_status_on_exception=False,
    ) as current_span:
        yield current_span


# ---------------------------------------------------------------------------
# Streaming span wrapper objects
# ---------------------------------------------------------------------------

class _StreamingSpanWrapper:
    """
    Wraps a synchronous SSE iterator and guarantees ``span.end()`` is called
    regardless of *how* the caller exits — normal exhaustion, ``break``, or
    garbage collection.

    Implements the full iterator protocol plus ``close()`` so callers that
    call ``generator.close()`` explicitly are handled correctly too.
    """

    def __init__(self, result: Iterator[Any], span: Span) -> None:
        self._result = result
        self._span = span
        self._last_artifact_type: Optional[str] = None
        self._last_task_state: Optional[str] = None
        self._span_ended = False

    # --- iterator protocol ---------------------------------------------------

    def __iter__(self) -> "_StreamingSpanWrapper":
        return self

    def __next__(self) -> Any:
        try:
            event = next(self._result)
        except StopIteration:
            self._finalize_span()
            raise
        except Exception as exc:
            self._span.record_exception(exc)
            self._span.set_status(
                Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}")
            )
            self._finalize_span(error=True)
            raise

        # Capture artifact type from streaming events
        artifact = getattr(event, "artifact", None)
        if artifact is not None:
            atype = get_artifact_type(artifact)
            if atype:
                self._last_artifact_type = atype

        # Capture task state from status events
        task_state = _extract_event_state(event)
        if task_state:
            self._last_task_state = task_state

        return event

    # --- explicit close (e.g. caller does ``for e in s: break``) ------------

    def close(self) -> None:
        try:
            if hasattr(self._result, "close"):
                self._result.close()
        finally:
            self._finalize_span()

    # --- last-resort gc finalizer -------------------------------------------

    def __del__(self) -> None:

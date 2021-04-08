"""Callback handler for Strands Agents instrumentation.

This module provides a callback handler that captures agent events and
converts them into OpenTelemetry spans, complementing Strands' built-in
telemetry with additional detail.
"""

import json
import time
from typing import Any, Callable, Dict, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, SpanKind

from traceai_strands._attributes import (
    SpanAttributes,
    get_agent_attributes,
    get_tool_attributes,
    get_model_provider,
)


def safe_serialize(obj: Any, max_length: int = 10000) -> str:
    """Safely serialize an object to a string.

    Args:
        obj: Object to serialize.
        max_length: Maximum length of the result.

    Returns:
        String representation of the object.
    """
    if obj is None:
        return "null"

    if isinstance(obj, str):
        result = obj
    elif isinstance(obj, (dict, list)):
        try:
            result = json.dumps(obj, default=str)
        except (TypeError, ValueError):
            result = str(obj)
    else:
        result = str(obj)

    if len(result) > max_length:
        return result[: max_length - 3] + "..."

    return result


class StrandsCallbackHandler:
    """Callback handler for Strands Agents that creates OpenTelemetry spans.

    This handler captures agent lifecycle events and converts them into
    spans, providing detailed observability beyond Strands' built-in
    telemetry.

    Example usage:

        from strands import Agent
        from traceai_strands import StrandsCallbackHandler

        callback = StrandsCallbackHandler(tracer_provider=trace_provider)
        agent = Agent(
            model="us.anthropic.claude-sonnet-4-20250514-v1:0",
            callback_handler=callback,
        )

        response = agent("What is the weather today?")
    """

    def __init__(
        self,
        tracer_provider: Optional[trace.TracerProvider] = None,
        capture_input: bool = True,
        capture_output: bool = True,
    ):
        """Initialize the callback handler.

        Args:
            tracer_provider: Optional OpenTelemetry tracer provider.
            capture_input: Whether to capture input prompts.
            capture_output: Whether to capture output responses.
        """
        if tracer_provider is None:
            tracer_provider = trace.get_tracer_provider()

        self._tracer = tracer_provider.get_tracer(
            "traceai_strands",
            schema_url="https://opentelemetry.io/schemas/1.11.0",
        )
        self._capture_input = capture_input
        self._capture_output = capture_output

        # Track active spans for correlation
        self._active_spans: Dict[str, trace.Span] = {}
        self._agent_span: Optional[trace.Span] = None
        self._start_time: Optional[float] = None

    def __call__(self, **kwargs) -> Any:
        """Handle callback events from Strands.

        Strands callbacks receive different kwargs based on the event type.
        Common patterns:
        - data: The event data
        - event: The event type name
        - agent: The agent instance

        Args:
            **kwargs: Event-specific keyword arguments.

        Returns:
            None or callback-specific return value.
        """
        event = kwargs.get("event", "unknown")
        data = kwargs.get("data", {})

        handler = getattr(self, f"_on_{event}", None)
        if handler:
            return handler(data, kwargs)

        # Default handling for unknown events
        return self._on_generic_event(event, data, kwargs)

    def on_agent_start(self, agent: Any, prompt: str) -> None:
        """Called when an agent starts processing.

        Args:
            agent: The Strands Agent instance.
            prompt: The user prompt.
        """
        self._start_time = time.time()

        attrs = get_agent_attributes(agent)
        attrs["gen_ai.operation.name"] = "agent.invoke"

        if self._capture_input:
            attrs["gen_ai.prompt"] = safe_serialize(prompt)

        self._agent_span = self._tracer.start_span(
            name="strands.agent",
            kind=SpanKind.CLIENT,
            attributes=attrs,
        )

    def on_agent_end(self, agent: Any, response: Any) -> None:
        """Called when an agent finishes processing.

        Args:
            agent: The Strands Agent instance.
            response: The agent's response.
        """
        if self._agent_span is None:
            return

        if self._capture_output:
            self._agent_span.set_attribute(
                "gen_ai.completion",
                safe_serialize(response),
            )

        # Calculate duration
        if self._start_time:
            duration = time.time() - self._start_time

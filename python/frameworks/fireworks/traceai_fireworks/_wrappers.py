"""Wrapper classes for Fireworks AI instrumentation."""

import logging
from abc import ABC
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Tuple

from fi_instrumentation import safe_json_dumps
from fi_instrumentation.fi_types import FiSpanKindValues, SpanAttributes, MessageAttributes
from opentelemetry import trace as trace_api
from opentelemetry.trace import INVALID_SPAN, Span, Status, StatusCode
from opentelemetry.util.types import AttributeValue

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Fireworks-specific base URL for detection
FIREWORKS_BASE_URL = "api.fireworks.ai"


class _WithTracer(ABC):
    """Base class for wrappers that need a tracer."""

    def __init__(self, tracer: trace_api.Tracer, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer

    @contextmanager
    def _start_span(
        self,
        span_name: str,
        attributes: Dict[str, AttributeValue],
    ) -> Iterator[Span]:
        """Start a span with the given name and attributes."""
        try:
            span = self._tracer.start_span(name=span_name, attributes=attributes)
        except Exception:
            span = INVALID_SPAN

        with trace_api.use_span(
            span,
            end_on_exit=False,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            yield span


def _is_fireworks_client(instance: Any) -> bool:
    """Check if the client instance is configured for Fireworks AI."""
    try:
        if hasattr(instance, "_client"):
            client = instance._client
            if hasattr(client, "base_url"):
                base_url = str(client.base_url)
                return FIREWORKS_BASE_URL in base_url
    except Exception:
        pass
    return False


def _get_input_attributes(
    model: str,
    messages: list,
    **kwargs: Any,
) -> Dict[str, AttributeValue]:
    """Extract input attributes from request parameters."""
    attributes: Dict[str, AttributeValue] = {
        SpanAttributes.GEN_AI_SPAN_KIND: FiSpanKindValues.LLM.value,
        SpanAttributes.GEN_AI_PROVIDER_NAME: "fireworks",
        SpanAttributes.GEN_AI_PROVIDER_NAME: "fireworks",
        SpanAttributes.GEN_AI_REQUEST_MODEL: model,
    }

    # Add invocation parameters
    invocation_params = {}
    if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
        invocation_params["max_tokens"] = kwargs["max_tokens"]
    if "temperature" in kwargs and kwargs["temperature"] is not None:
        invocation_params["temperature"] = kwargs["temperature"]
    if "top_p" in kwargs and kwargs["top_p"] is not None:
        invocation_params["top_p"] = kwargs["top_p"]
    if "stop" in kwargs and kwargs["stop"] is not None:
        invocation_params["stop"] = kwargs["stop"]
    if "frequency_penalty" in kwargs and kwargs["frequency_penalty"] is not None:
        invocation_params["frequency_penalty"] = kwargs["frequency_penalty"]
    if "presence_penalty" in kwargs and kwargs["presence_penalty"] is not None:
        invocation_params["presence_penalty"] = kwargs["presence_penalty"]

    if invocation_params:
        attributes[SpanAttributes.GEN_AI_REQUEST_PARAMETERS] = safe_json_dumps(invocation_params)

    # Add input messages
    for i, message in enumerate(messages):
        prefix = f"{SpanAttributes.GEN_AI_INPUT_MESSAGES}.{i}"
        if isinstance(message, dict):
            if "role" in message:
                attributes[f"{prefix}.{MessageAttributes.MESSAGE_ROLE}"] = message["role"]
            if "content" in message:
                attributes[f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}"] = str(message["content"]) if message["content"] else ""
        else:
            # Handle message objects
            if hasattr(message, "role"):
                attributes[f"{prefix}.{MessageAttributes.MESSAGE_ROLE}"] = message.role
            if hasattr(message, "content"):
                attributes[f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}"] = str(message.content) if message.content else ""

    # Add raw input
    raw_input = {"model": model, "messages": messages}
    raw_input.update({k: v for k, v in kwargs.items() if v is not None})
    attributes[SpanAttributes.INPUT_VALUE] = safe_json_dumps(raw_input)

    return attributes


def _get_output_attributes(response: Any) -> Dict[str, AttributeValue]:
    """Extract output attributes from response."""
    attributes: Dict[str, AttributeValue] = {}

    # Get model from response
    if hasattr(response, "model"):
        attributes[SpanAttributes.GEN_AI_REQUEST_MODEL] = response.model

    # Get choices/messages
    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        prefix = f"{SpanAttributes.GEN_AI_OUTPUT_MESSAGES}.0"

        if hasattr(choice, "message"):
            message = choice.message
            if hasattr(message, "role"):
                attributes[f"{prefix}.{MessageAttributes.MESSAGE_ROLE}"] = message.role
            if hasattr(message, "content") and message.content:
                attributes[f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}"] = message.content

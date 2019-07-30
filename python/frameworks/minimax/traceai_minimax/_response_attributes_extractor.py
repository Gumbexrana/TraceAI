import logging
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

from fi_instrumentation.fi_types import (
    MessageAttributes,
    SpanAttributes,
)
from opentelemetry.util.types import AttributeValue

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _ChatCompletionResponseAttributesExtractor:
    """Extract span attributes from MiniMax chat completion response."""

    def get_attributes(
        self,
        response: Optional[Dict[str, Any]],
        is_streaming: bool = False,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        """Extract primary attributes from response."""
        if response is None:
            return

        # Token usage
        usage = response.get("usage", {})
        if usage:
            if "prompt_tokens" in usage:
                yield SpanAttributes.GEN_AI_USAGE_INPUT_TOKENS, usage["prompt_tokens"]
            if "completion_tokens" in usage:
                yield SpanAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, usage["completion_tokens"]
            if "total_tokens" in usage:
                yield SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS, usage["total_tokens"]

        # Model info
        if "model" in response:
            yield SpanAttributes.GEN_AI_REQUEST_MODEL, response["model"]

        # Response ID
        if "id" in response:
            yield "minimax.response_id", response["id"]

    def get_extra_attributes(
        self,
        response: Optional[Dict[str, Any]],
        request_parameters: Mapping[str, Any],
        is_streaming: bool = False,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        """Extract extra attributes including output."""
        if response is None:
            return

        choices = response.get("choices", [])

        for i, choice in enumerate(choices):
            message = choice.get("message", {})

            # Role
            role = message.get("role", "assistant")
            yield f"{SpanAttributes.GEN_AI_OUTPUT_MESSAGES}.{i}.{MessageAttributes.MESSAGE_ROLE}", role

            # Content
            content = message.get("content", "")
            if content:
                yield f"{SpanAttributes.GEN_AI_OUTPUT_MESSAGES}.{i}.{MessageAttributes.MESSAGE_CONTENT}", content

            # Tool calls
            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                yield "minimax.tool_calls_count", len(tool_calls)
                for j, tc in enumerate(tool_calls[:10]):  # Limit
                    yield f"minimax.tool_calls.{j}.id", tc.get("id", "")
                    yield f"minimax.tool_calls.{j}.type", tc.get("type", "function")
                    if "function" in tc:
                        yield f"minimax.tool_calls.{j}.function.name", tc["function"].get("name", "")

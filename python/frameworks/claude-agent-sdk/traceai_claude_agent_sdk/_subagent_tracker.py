"""Subagent tracking for Claude Agent SDK instrumentation.

This module provides tracking for subagent (Task tool) executions,
maintaining proper parent-child span relationships and aggregating
costs across the subagent hierarchy.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from opentelemetry import trace as trace_api
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from ._attributes import ClaudeAgentAttributes, ClaudeAgentSpanKind

logger = logging.getLogger(__name__)

# Span name for subagent executions
SUBAGENT_SPAN_NAME = "claude_agent.subagent"


@dataclass
class SubagentInfo:
    """Information about a running subagent."""

    tool_use_id: str
    span: Span
    start_time: float
    subagent_type: Optional[str] = None
    description: Optional[str] = None
    prompt: Optional[str] = None
    parent_tool_use_id: Optional[str] = None

    # Aggregated metrics
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    tool_calls: int = 0
    turns: int = 0

    # Child subagents for hierarchy
    child_subagent_ids: List[str] = field(default_factory=list)


class SubagentTracker:
    """Track subagent execution hierarchy.

    This class manages the lifecycle of subagent spans, ensuring proper
    parent-child relationships and aggregating costs across the hierarchy.

    Usage:
        tracker = SubagentTracker(tracer)

        # When Task tool is invoked
        span = tracker.start_subagent(tool_use_id, input_data, parent_span)

        # When subagent completes
        tracker.end_subagent(tool_use_id, result)

        # Get aggregated costs
        costs = tracker.get_hierarchy_costs(root_tool_use_id)
    """

    def __init__(self, tracer: trace_api.Tracer):
        """Initialize the subagent tracker.

        Args:
            tracer: OpenTelemetry tracer for creating spans
        """
        self.tracer = tracer
        self._active_subagents: Dict[str, SubagentInfo] = {}
        self._completed_subagents: Dict[str, SubagentInfo] = {}

    def start_subagent(
        self,
        tool_use_id: str,
        input_data: Dict[str, Any],
        parent_span: Optional[Span] = None,
        parent_tool_use_id: Optional[str] = None,
    ) -> Span:
        """Start tracking a new subagent.

        Args:
            tool_use_id: Unique ID for this Task tool use
            input_data: Task tool input containing subagent details
            parent_span: Parent span to link this subagent to
            parent_tool_use_id: ID of parent Task tool (for nested subagents)

        Returns:
            The created subagent span
        """
        # Extract subagent details from input
        subagent_type = input_data.get("subagent_type", "unknown")
        description = input_data.get("description", "")
        prompt = input_data.get("prompt", "")
        allowed_tools = input_data.get("allowed_tools", [])
        model = input_data.get("model")

        # Build span attributes
        attributes = {
            ClaudeAgentAttributes.SPAN_KIND: ClaudeAgentSpanKind.SUBAGENT.value,
            ClaudeAgentAttributes.TOOL_USE_ID: tool_use_id,
            ClaudeAgentAttributes.SUBAGENT_TYPE: subagent_type,
        }

        if description:
            attributes[ClaudeAgentAttributes.SUBAGENT_DESCRIPTION] = description[:500]
        if prompt:
            attributes[ClaudeAgentAttributes.SUBAGENT_PROMPT] = prompt[:2000]
        if allowed_tools:
            attributes[ClaudeAgentAttributes.SUBAGENT_TOOLS] = json.dumps(allowed_tools)
        if model:
            attributes[ClaudeAgentAttributes.AGENT_MODEL] = model
        if parent_tool_use_id:
            attributes[ClaudeAgentAttributes.PARENT_TOOL_USE_ID] = parent_tool_use_id

        # Create span as child of parent
        context = None
        if parent_span:
            context = trace_api.set_span_in_context(parent_span)

        span = self.tracer.start_span(
            name=f"{SUBAGENT_SPAN_NAME}.{subagent_type}",
            kind=SpanKind.INTERNAL,
            context=context,
            attributes=attributes,
        )

        # Track the subagent
        info = SubagentInfo(
            tool_use_id=tool_use_id,
            span=span,
            start_time=time.time(),
            subagent_type=subagent_type,
            description=description,
            prompt=prompt,
            parent_tool_use_id=parent_tool_use_id,
        )
        self._active_subagents[tool_use_id] = info

        # Link to parent subagent if nested
        if parent_tool_use_id and parent_tool_use_id in self._active_subagents:
            parent_info = self._active_subagents[parent_tool_use_id]
            parent_info.child_subagent_ids.append(tool_use_id)

        logger.debug(f"Started subagent {subagent_type} with ID {tool_use_id}")

        return span

    def end_subagent(
        self,
        tool_use_id: str,
        result: Optional[Any] = None,
        error: Optional[Exception] = None,
        usage: Optional[Dict[str, Any]] = None,
    ) -> Optional[SubagentInfo]:
        """End a subagent and record results.

        Args:
            tool_use_id: ID of the Task tool use
            result: Subagent result/output
            error: Exception if subagent failed
            usage: Usage metrics from subagent

        Returns:
            SubagentInfo with final metrics, or None if not found
        """
        info = self._active_subagents.pop(tool_use_id, None)
        if not info:
            logger.warning(f"No active subagent found for ID {tool_use_id}")
            return None

        span = info.span

        # Record duration
        duration_ms = (time.time() - info.start_time) * 1000
        span.set_attribute(ClaudeAgentAttributes.TOOL_DURATION_MS, duration_ms)

        # Record usage if provided
        if usage:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            total_cost = usage.get("total_cost_usd", 0.0)

            info.total_input_tokens = input_tokens
            info.total_output_tokens = output_tokens
            info.total_cost_usd = total_cost

            span.set_attribute(ClaudeAgentAttributes.USAGE_INPUT_TOKENS, input_tokens)
            span.set_attribute(ClaudeAgentAttributes.USAGE_OUTPUT_TOKENS, output_tokens)
            if total_cost:
                span.set_attribute(ClaudeAgentAttributes.COST_TOTAL_USD, total_cost)

        # Aggregate child subagent costs
        child_costs = self._aggregate_child_costs(tool_use_id)
        if child_costs:
            info.total_input_tokens += child_costs["input_tokens"]
            info.total_output_tokens += child_costs["output_tokens"]
            info.total_cost_usd += child_costs["total_cost_usd"]

            span.set_attribute(
                "claude_agent.subagent.aggregated_input_tokens",
                info.total_input_tokens,
            )
            span.set_attribute(
                "claude_agent.subagent.aggregated_output_tokens",
                info.total_output_tokens,
            )
            span.set_attribute(
                "claude_agent.subagent.aggregated_cost_usd",
                info.total_cost_usd,
            )

        # Record result
        if result is not None:
            result_str = str(result)[:2000] if result else ""
            span.set_attribute(ClaudeAgentAttributes.TOOL_OUTPUT, result_str)

        # Handle error
        if error:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
            span.set_attribute(ClaudeAgentAttributes.TOOL_IS_ERROR, True)
            span.set_attribute(ClaudeAgentAttributes.TOOL_ERROR_MESSAGE, str(error))
        else:
            span.set_status(Status(StatusCode.OK))
            span.set_attribute(ClaudeAgentAttributes.TOOL_IS_ERROR, False)

        span.end()

        # Move to completed
        self._completed_subagents[tool_use_id] = info

        logger.debug(f"Ended subagent {info.subagent_type} with ID {tool_use_id}")

        return info

    def record_subagent_turn(self, tool_use_id: str) -> None:
        """Record a turn within a subagent.

        Args:
            tool_use_id: ID of the subagent
        """
        if tool_use_id in self._active_subagents:
            self._active_subagents[tool_use_id].turns += 1

    def record_subagent_tool_call(self, tool_use_id: str) -> None:
        """Record a tool call within a subagent.

        Args:
            tool_use_id: ID of the subagent
        """
        if tool_use_id in self._active_subagents:
            self._active_subagents[tool_use_id].tool_calls += 1

    def get_active_subagent(self, tool_use_id: str) -> Optional[SubagentInfo]:
        """Get info about an active subagent.

"""Session tracking for Claude Agent SDK instrumentation.

This module provides tracking for session continuity across
query invocations, supporting resume and fork operations with
proper trace linking.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from opentelemetry import trace as trace_api
from opentelemetry.trace import Span, SpanKind, Link, Status, StatusCode
from opentelemetry.trace.propagation import get_current_span

from ._attributes import ClaudeAgentAttributes, ClaudeAgentSpanKind

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Information about a session."""

    session_id: str
    created_at: float
    is_new: bool = True
    previous_session_id: Optional[str] = None
    fork_from_session_id: Optional[str] = None

    # Trace context for linking
    trace_id: Optional[str] = None
    span_id: Optional[str] = None

    # Aggregated metrics across the session
    total_queries: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_turns: int = 0


class SessionTracker:
    """Track session continuity across queries.

    This class manages session state and creates proper trace links
    when sessions are resumed or forked.

    Usage:
        tracker = SessionTracker(tracer)

        # On new session
        tracker.start_session(session_id, span)

        # On resume
        tracker.resume_session(session_id, previous_session_id, span)

        # On fork
        tracker.fork_session(new_session_id, fork_from_session_id, span)

        # Get session info
        info = tracker.get_session(session_id)
    """

    def __init__(self, tracer: trace_api.Tracer):
        """Initialize the session tracker.

        Args:
            tracer: OpenTelemetry tracer for creating spans
        """
        self.tracer = tracer
        self._sessions: Dict[str, SessionInfo] = {}
        self._active_session_id: Optional[str] = None

    def start_session(
        self,
        session_id: str,
        span: Span,
        is_resumed: bool = False,
        previous_session_id: Optional[str] = None,
    ) -> SessionInfo:
        """Start tracking a new or resumed session.

        Args:
            session_id: Unique session identifier
            span: The conversation span to annotate
            is_resumed: Whether this is a resumed session
            previous_session_id: ID of the previous session (for resume)

        Returns:
            SessionInfo for the session
        """
        # Check if we're resuming an existing session
        if session_id in self._sessions:
            info = self._sessions[session_id]
            info.total_queries += 1
            logger.debug(f"Continuing session {session_id}")
        else:
            info = SessionInfo(
                session_id=session_id,
                created_at=time.time(),
                is_new=not is_resumed,
                previous_session_id=previous_session_id,
            )
            info.total_queries = 1
            self._sessions[session_id] = info
            logger.debug(f"Started new session {session_id}")

        # Store trace context for future linking
        span_context = span.get_span_context()
        if span_context.is_valid:
            info.trace_id = format(span_context.trace_id, "032x")
            info.span_id = format(span_context.span_id, "016x")

        # Set session attributes on span
        span.set_attribute(ClaudeAgentAttributes.GEN_AI_CONVERSATION_ID, session_id)
        span.set_attribute(ClaudeAgentAttributes.SESSION_IS_NEW, info.is_new)
        span.set_attribute(ClaudeAgentAttributes.SESSION_IS_RESUMED, is_resumed)

        if previous_session_id:
            span.set_attribute(
                ClaudeAgentAttributes.SESSION_PREVIOUS_ID,
                previous_session_id,
            )

        self._active_session_id = session_id

        return info

    def resume_session(
        self,
        session_id: str,
        previous_session_id: str,
        span: Span,
    ) -> SessionInfo:
        """Resume a previous session with trace linking.

        Args:
            session_id: New session identifier
            previous_session_id: ID of the session being resumed
            span: The conversation span to annotate

        Returns:
            SessionInfo for the resumed session
        """
        # Get previous session info for linking
        previous_info = self._sessions.get(previous_session_id)

        # Create links to previous session if we have trace context
        if previous_info and previous_info.trace_id and previous_info.span_id:
            # Note: In a real implementation, we would create a Link here
            # For now, we store the reference in attributes
            span.set_attribute(
                "claude_agent.session.previous_trace_id",
                previous_info.trace_id,
            )
            span.set_attribute(
                "claude_agent.session.previous_span_id",
                previous_info.span_id,
            )

        # Start the new session as resumed
        info = self.start_session(
            session_id=session_id,
            span=span,
            is_resumed=True,
            previous_session_id=previous_session_id,
        )

        # Carry over aggregated metrics from previous session
        if previous_info:
            info.total_input_tokens = previous_info.total_input_tokens
            info.total_output_tokens = previous_info.total_output_tokens
            info.total_cost_usd = previous_info.total_cost_usd
            info.total_turns = previous_info.total_turns
            # Note: total_queries starts fresh for the new session

        logger.debug(f"Resumed session {previous_session_id} as {session_id}")

        return info

    def fork_session(
        self,
        new_session_id: str,
        fork_from_session_id: str,
        span: Span,
    ) -> SessionInfo:
        """Fork a session to create a branch.

        Args:
            new_session_id: New session identifier
            fork_from_session_id: ID of the session being forked
            span: The conversation span to annotate

        Returns:
            SessionInfo for the forked session
        """
        # Get source session info
        source_info = self._sessions.get(fork_from_session_id)

        # Create new session as a fork
        info = SessionInfo(
            session_id=new_session_id,
            created_at=time.time(),
            is_new=True,  # Fork creates a new branch
            fork_from_session_id=fork_from_session_id,
        )
        info.total_queries = 1
        self._sessions[new_session_id] = info

        # Store trace context
        span_context = span.get_span_context()
        if span_context.is_valid:
            info.trace_id = format(span_context.trace_id, "032x")
            info.span_id = format(span_context.span_id, "016x")

        # Set fork attributes
        span.set_attribute(ClaudeAgentAttributes.GEN_AI_CONVERSATION_ID, new_session_id)
        span.set_attribute(ClaudeAgentAttributes.SESSION_IS_NEW, True)
        span.set_attribute(ClaudeAgentAttributes.SESSION_FORK_FROM, fork_from_session_id)

        # Link to source session
        if source_info and source_info.trace_id and source_info.span_id:
            span.set_attribute(
                "claude_agent.session.fork_source_trace_id",
                source_info.trace_id,
            )
            span.set_attribute(
                "claude_agent.session.fork_source_span_id",
                source_info.span_id,
            )

        self._active_session_id = new_session_id

        logger.debug(f"Forked session {fork_from_session_id} to {new_session_id}")

        return info

    def update_session_metrics(
        self,
        session_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        turns: int = 0,
    ) -> None:
        """Update aggregated metrics for a session.

        Args:
            session_id: Session identifier

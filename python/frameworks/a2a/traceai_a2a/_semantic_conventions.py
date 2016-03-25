"""
A2A Protocol semantic conventions for traceAI.

These constants define the OpenTelemetry span attribute names used when
instrumenting Google's Agent-to-Agent (A2A) Protocol. They extend the
gen_ai.* namespace used by the rest of traceAI for consistency.

Reference: https://google.github.io/A2A/
"""

# Task attributes
A2A_TASK_ID = "gen_ai.a2a.task.id"
"""Unique identifier for an A2A task (UUID string)."""

A2A_TASK_STATE = "gen_ai.a2a.task.state"
"""
Current state of the A2A task.
One of: submitted | working | input-required | completed | failed | canceled
"""

# Agent identification attributes
A2A_AGENT_URL = "gen_ai.a2a.agent.url"
"""Base URL of the remote A2A agent endpoint being called."""

A2A_AGENT_CARD_NAME = "gen_ai.a2a.agent.card.name"
"""Human-readable name from the remote agent's AgentCard discovery document."""

A2A_AGENT_CARD_VERSION = "gen_ai.a2a.agent.card.version"
"""Version string from the remote agent's AgentCard discovery document."""

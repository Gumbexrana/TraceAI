"""
Attribute extractors for A2A Protocol objects.

These functions extract span attributes from A2A SDK model objects
(Task, Message, Artifact, AgentCard) in a safe, exception-guarded way —
following the same pattern as other traceAI attribute extractors.
"""

import json
import logging
from typing import Any, Dict, Iterator, Optional, Tuple

from traceai_a2a._semantic_conventions import (
    A2A_AGENT_CARD_NAME,
    A2A_AGENT_CARD_VERSION,
    A2A_AGENT_URL,
    A2A_ARTIFACT_TYPE,
    A2A_MESSAGE_PARTS_COUNT,
    A2A_MESSAGE_ROLE,
    A2A_PUSH_NOTIFICATION_URL,
    A2A_STREAMING,
    A2A_TASK_ID,
    A2A_TASK_STATE,
)

logger = logging.getLogger(__name__)

AttributeYield = Iterator[Tuple[str, Any]]


def get_task_attributes(task: Any) -> AttributeYield:
    """
    Extract span attributes from an A2A Task object (or task dict payload).

    Handles both A2A SDK model objects (with .id, .status attributes) and
    raw dict payloads (passed before a task object is returned).
    """
    try:
        if hasattr(task, "id") and task.id:
            yield A2A_TASK_ID, str(task.id)
        elif isinstance(task, dict) and task.get("id"):
            yield A2A_TASK_ID, str(task["id"])
    except Exception:
        logger.debug("Failed to extract task ID", exc_info=True)

    try:
        # Handle A2A SDK TaskStatus object
        status = getattr(task, "status", None)
        if status is not None:
            state = getattr(status, "state", None)
            if state is not None:
                # state may be an Enum or a plain string
                yield A2A_TASK_STATE, _enum_or_str(state)
        elif isinstance(task, dict):
            state = task.get("status", {}).get("state")
            if state:
                yield A2A_TASK_STATE, str(state)
    except Exception:
        logger.debug("Failed to extract task state", exc_info=True)


def get_message_attributes(message: Any) -> AttributeYield:
    """
    Extract span attributes from an A2A Message object or dict.
    """
    try:

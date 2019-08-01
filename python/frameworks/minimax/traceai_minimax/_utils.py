import logging
from enum import Enum
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple
from urllib.parse import urlparse

from fi_instrumentation import safe_json_dumps
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue

from traceai_minimax._with_span import _WithSpan

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class MiniMaxBaseURL(str, Enum):
    """MiniMax API base URL hostnames."""
    API_IO = "api.minimax.io"
    API_CHAT = "api.minimax.chat"


def is_minimax_client(instance: Any) -> bool:
    """Check if the OpenAI client is configured for MiniMax."""
    try:
        base_url = getattr(instance, "base_url", None)
        if base_url is None:
            # Check _client for wrapped clients
            client = getattr(instance, "_client", None)
            if client is not None:
                base_url = getattr(client, "base_url", None)

        if base_url is not None:
            parsed = urlparse(str(base_url))
            hostname = (parsed.hostname or "").lower()
            return hostname in {url.value for url in MiniMaxBaseURL}
    except Exception:

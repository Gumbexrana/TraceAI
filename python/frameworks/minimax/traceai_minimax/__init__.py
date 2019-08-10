import logging
from importlib import import_module
from typing import Any, Collection

logger = logging.getLogger(__name__)

from fi_instrumentation import FITracer, TraceConfig
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from wrapt import wrap_function_wrapper

from traceai_minimax._wrappers import (
    _AsyncChatCompletionWrapper,
    _ChatCompletionWrapper,
)
from traceai_minimax.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_instruments = ("openai >= 1.0.0",)


class MiniMaxInstrumentor(BaseInstrumentor):
    """
    OpenTelemetry instrumentor for MiniMax.

    MiniMax uses the OpenAI SDK with a custom base_url (https://api.minimax.io/v1).
    This instrumentor wraps OpenAI client methods and only instruments calls
    when the client is configured with a MiniMax base URL.

    Supports:
    - MiniMax-M2.7 (latest flagship model with enhanced reasoning and coding)
    - MiniMax-M2.7-highspeed (high-speed version of M2.7 for low-latency scenarios)
    - MiniMax-M2.5 (204K context window)
    - MiniMax-M2.5-highspeed (204K context window, faster inference)
    - Streaming and non-streaming responses
    - Function/tool calling
    """

    __slots__ = (
        "_original_chat_create",

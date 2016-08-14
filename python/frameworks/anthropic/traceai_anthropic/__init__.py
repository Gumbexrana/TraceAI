import logging
from typing import Any, Collection

from fi_instrumentation import FITracer
from fi_instrumentation.instrumentation._protect_wrapper import GuardrailProtectWrapper
from fi_instrumentation.instrumentation.config import TraceConfig
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import (
    BaseInstrumentor,
)  # type: ignore[attr-defined]
from traceai_anthropic._wrappers import (
    _AsyncCompletionsWrapper,
    _AsyncMessagesWrapper,
    _CompletionsWrapper,
    _MessagesCountTokensWrapper,
    _MessagesWrapper,
)
from traceai_anthropic.version import __version__
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("anthropic >= 0.30.0",)


class AnthropicInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """An instrumentor for the Anthropic framework."""

    __slots__ = (
        "_original_completions_create",
        "_original_async_completions_create",
        "_original_messages_create",
        "_original_async_messages_create",
        "_original_protect",
        "_instruments",
        "_tracer",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        from anthropic.resources.completions import AsyncCompletions, Completions
        from anthropic.resources.messages import AsyncMessages, Messages
        try:
            from fi.evals import Protect
        except ImportError:
            logger.warning("ai-evaluation is not installed, please install it to trace protect")
            Protect = None


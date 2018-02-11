import logging
from typing import Any, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.trace import get_tracer, get_tracer_provider
from wrapt import wrap_function_wrapper

from fi_instrumentation import FITracer, TraceConfig
from traceai_google_genai.package import _instruments
from traceai_google_genai.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class GoogleGenAIInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for `google-genai`
    """

    __slots__ = (
        "_original_generate_content",
        "_original_async_generate_content",
        "_original_generate_content_stream",
        "_original_async_generate_content_stream",
        "_tracer",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = get_tracer_provider()
        if not (config := kwargs.get("config")):
            config = TraceConfig()
        else:
            assert isinstance(config, TraceConfig)
        self._tracer = FITracer(
            get_tracer(__name__, __version__, tracer_provider),

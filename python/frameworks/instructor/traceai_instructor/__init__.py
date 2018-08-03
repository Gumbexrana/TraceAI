from importlib import import_module
from typing import Any, Collection
import logging
logger = logging.getLogger(__name__)
try:
    from fi.evals import Protect
except ImportError:
    logger.warning("ai-evaluation is not installed, please install it to trace protect")
    Protect = None
    pass
from fi_instrumentation import FITracer, TraceConfig
from fi_instrumentation.instrumentation._protect_wrapper import GuardrailProtectWrapper
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from traceai_instructor._wrappers import _HandleResponseWrapper, _PatchWrapper
from traceai_instructor.version import __version__
from wrapt import wrap_function_wrapper

_instruments = ("instructor >= 1.0.0",)


class InstructorInstrumentor(BaseInstrumentor):  # type: ignore
    __slots__ = (
        "_tracer",
        "_original_create",
        "_original_async_create",
        "_original_patch",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):

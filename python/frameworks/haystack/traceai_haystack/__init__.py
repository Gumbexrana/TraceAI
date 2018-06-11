import logging
from typing import Any, Callable, Collection
logger = logging.getLogger(__name__)
import haystack
try:
    from fi.evals import Protect
except ImportError:
    logger.warning("ai-evaluation is not installed, please install it to trace protect")
    Protect = None
    pass
from fi_instrumentation import FITracer, TraceConfig
from fi_instrumentation.instrumentation._protect_wrapper import GuardrailProtectWrapper
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import (
    BaseInstrumentor,
)  # type: ignore[attr-defined]
from traceai_haystack._wrappers import (
    _ComponentRunWrapper,
    _PipelineRunComponentWrapper,
    _PipelineWrapper,
)
from traceai_haystack.version import __version__
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("haystack-ai >= 2.9.0",)


class HaystackInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """An instrumentor for the Haystack framework."""

    __slots__ = (
        "_original_pipeline_run",
        "_original_pipeline_run_component",
        "_original_component_run_methods",
        "_tracer",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments


import logging
from importlib import import_module
from typing import Any, Collection

from fi_instrumentation import FITracer, TraceConfig
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from traceai_crewai._wrappers import (
    _ExecuteCoreWrapper,
    _KickoffWrapper,
    _ToolUseWrapper,
)
from traceai_crewai.version import __version__
from wrapt import wrap_function_wrapper

_instruments = ("crewai >= 0.41.1",)

logger = logging.getLogger(__name__)


class CrewAIInstrumentor(BaseInstrumentor):  # type: ignore
    __slots__ = (
        "_original_execute_core",
        "_original_kickoff",
        "_original_tool_use",
        "_tracer",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        if not (config := kwargs.get("config")):
            config = TraceConfig()
        else:
            assert isinstance(config, TraceConfig)
        self._tracer = FITracer(
            trace_api.get_tracer(__name__, __version__, tracer_provider),
            config=config,
        )

        execute_core_wrapper = _ExecuteCoreWrapper(tracer=self._tracer)
        self._original_execute_core = getattr(
            import_module("crewai").Task, "_execute_core", None
        )
        wrap_function_wrapper(
            module="crewai",
            name="Task._execute_core",
            wrapper=execute_core_wrapper,
        )

        kickoff_wrapper = _KickoffWrapper(tracer=self._tracer)
        self._original_kickoff = getattr(import_module("crewai").Crew, "kickoff", None)
        wrap_function_wrapper(
            module="crewai",
            name="Crew.kickoff",
            wrapper=kickoff_wrapper,
        )

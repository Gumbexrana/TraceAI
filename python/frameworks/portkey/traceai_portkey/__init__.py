import logging
from importlib import import_module
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from portkey_ai.api_resources.apis.chat_complete import Completions
from wrapt import wrap_function_wrapper

from fi_instrumentation import FITracer, TraceConfig
from traceai_portkey._wrappers import _CompletionsWrapper
from traceai_portkey.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_instruments = ("portkey_ai >= 0.1.0",)


class PortkeyInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """An instrumentor for the Portkey AI framework."""


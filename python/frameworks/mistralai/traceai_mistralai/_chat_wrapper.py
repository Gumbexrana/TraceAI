import json
import logging
import warnings
from abc import ABC
from contextlib import contextmanager
from inspect import Signature, signature
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Tuple,
)

from fi_instrumentation import get_attributes_from_context, safe_json_dumps
from fi_instrumentation.fi_types import (
    FiMimeTypeValues,
    FiSpanKindValues,
    SpanAttributes,
)
from mistralai.utils.eventstreaming import EventStream, EventStreamAsync
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import INVALID_SPAN
from opentelemetry.util.types import AttributeValue
from traceai_mistralai._request_attributes_extractor import _RequestAttributesExtractor
from traceai_mistralai._response_accumulator import _ChatCompletionAccumulator
from traceai_mistralai._response_attributes_extractor import (
    _ResponseAttributesExtractor,
    _StreamResponseAttributesExtractor,
)
from traceai_mistralai._stream import _AsyncStream, _Stream
from traceai_mistralai._utils import (
    _finish_tracing,
    _process_input_messages,
    _raw_input,
)
from traceai_mistralai._with_span import _WithSpan

if TYPE_CHECKING:
    from mistralai import Mistral

__all__ = ("_SyncChatWrapper",)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _WithTracer(ABC):
    def __init__(self, tracer: trace_api.Tracer, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer

    @contextmanager
    def _start_as_current_span(
        self,
        span_name: str,
        attributes: Iterable[Tuple[str, AttributeValue]],
        context_attributes: Iterable[Tuple[str, AttributeValue]],
        extra_attributes: Iterable[Tuple[str, AttributeValue]],
    ) -> Iterator[_WithSpan]:
        # Because OTEL has a default limit of 128 attributes, we split our
        # attributes into two tiers, where "extra_attributes" are added first to
        # ensure that the most important "attributes" are added last and are not
        # dropped.
        try:
            span = self._tracer.start_span(
                name=span_name, attributes=dict(extra_attributes)
            )
        except Exception:
            logger.exception("Failed to start span")
            span = INVALID_SPAN
        with trace_api.use_span(
            span,
            end_on_exit=False,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            yield _WithSpan(
                span=span,
                context_attributes=dict(context_attributes),
                extra_attributes=dict(attributes),
            )


class _WithMistralAI(ABC):
    __slots__ = (
        "_request_attributes_extractor",
        "_response_attributes_extractor",
    )

    def __init__(self, mistralai: ModuleType, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._request_attributes_extractor = _RequestAttributesExtractor(mistralai)
        self._response_attributes_extractor = _ResponseAttributesExtractor()

    def _get_span_kind(self) -> str:
        return FiSpanKindValues.LLM.value

    def _get_attributes_from_request(
        self,
        request_parameters: Dict[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield SpanAttributes.GEN_AI_SPAN_KIND, self._get_span_kind()
        try:
            yield from _process_input_messages(request_parameters.get("messages", []))
            yield from _raw_input(request_parameters)
            # yield from _as_input_attributes(_io_value_and_type(request_parameters))
        except Exception:
            logger.exception(
                f"Failed to get input attributes from request parameters of "
                f"type {type(request_parameters)}"
            )

    def _get_extra_attributes_from_request(
        self,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        try:
            yield from self._request_attributes_extractor.get_attributes_from_request(
                request_parameters=request_parameters,
            )
        except Exception:
            logger.exception(
                f"Failed to get extra attributes from request options of "
                f"type {type(request_parameters)}"
            )

    def _parse_args(
        self,
        signature: Signature,
        mistral_client: "Mistral",
        *args: Tuple[Any],
        **kwargs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Serialize parameters to JSON.

        Based off of https://github.com/mistralai/client-python/blob/80c7951bad83338641d5e89684f841ce1cac938f/src/mistralai/client_base.py#L76
        """
        bound_signature = signature.bind(*args, **kwargs)
        bound_signature.apply_defaults()
        bound_arguments = bound_signature.arguments
        request_data: Dict[str, Any] = {}
        for key, value in bound_arguments.items():
            try:
                if value is not None:
                    try:
                        # ensure the value is JSON-serializable
                        safe_json_dumps(value)
                        request_data[key] = value
                    except json.JSONDecodeError:
                        request_data[key] = str(value)
            except Exception:
                request_data[key] = str(value)
        return request_data

    def _finalize_response(
        self,
        response: Any,
        with_span: _WithSpan,
        request_parameters: Mapping[str, Any],

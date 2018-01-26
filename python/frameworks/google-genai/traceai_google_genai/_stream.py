from collections import defaultdict
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue
from wrapt import ObjectProxy

from fi_instrumentation import safe_json_dumps
from traceai_google_genai._utils import (
    _as_output_attributes,
    _finish_tracing,
    _ValueAndType,
)
from traceai_google_genai._with_span import _WithSpan
from fi_instrumentation.fi_types import (
    MessageAttributes,
    FiMimeTypeValues,
    SpanAttributes,
    ToolCallAttributes,
)

if TYPE_CHECKING:
    from google.genai.types import GenerateContentResponse


class _Stream(ObjectProxy):  # type: ignore
    __slots__ = (
        "_response_accumulator",
        "_with_span",
        "_is_finished",
    )

    def __init__(
        self,
        stream: Iterator["GenerateContentResponse"],
        with_span: _WithSpan,
    ) -> None:
        super().__init__(stream)
        self._response_accumulator = _ResponseAccumulator()
        self._with_span = with_span

    def __iter__(self) -> Iterator["GenerateContentResponse"]:
        try:
            for item in self.__wrapped__:
                self._response_accumulator.process_chunk(item)
                yield item
        except Exception as exception:
            status = trace_api.Status(
                status_code=trace_api.StatusCode.ERROR,
                description=f"{type(exception).__name__}: {exception}",
            )
            self._with_span.record_exception(exception)
            self._finish_tracing(status=status)
            raise
        # completed without exception
        status = trace_api.Status(
            status_code=trace_api.StatusCode.OK,
        )
        self._finish_tracing(status=status)

    async def __aiter__(self) -> AsyncIterator["GenerateContentResponse"]:
        try:
            async for item in self.__wrapped__:
                self._response_accumulator.process_chunk(item)
                yield item
        except Exception as exception:
            status = trace_api.Status(
                status_code=trace_api.StatusCode.ERROR,
                description=f"{type(exception).__name__}: {exception}",
            )
            self._with_span.record_exception(exception)
            self._finish_tracing(status=status)
            raise
        # completed without exception
        status = trace_api.Status(
            status_code=trace_api.StatusCode.OK,
        )
        self._finish_tracing(status=status)

    def _finish_tracing(
        self,
        status: Optional[trace_api.Status] = None,
    ) -> None:
        response_extractor = _ResponseExtractor(response_accumulator=self._response_accumulator)
        _finish_tracing(
            with_span=self._with_span,
            attributes=response_extractor.get_attributes(),
            extra_attributes=response_extractor.get_extra_attributes(),
            status=status,
        )


class _ResponseAccumulator:
    __slots__ = (
        "_is_null",
        "_values",
    )

    def __init__(self) -> None:
        self._is_null = True
        self._values = _ValuesAccumulator(
            candidates=_IndexedAccumulator(
                lambda: _ValuesAccumulator(
                    content=_ValuesAccumulator(
                        parts=_IndexedAccumulator(
                            lambda: _ValuesAccumulator(
                                text=_StringAccumulator(),
                                function_call=_DictReplace(),
                            ),
                        ),
                        role=_SimpleStringReplace(),
                    ),
                    finish_reason=_SimpleStringReplace(),
                ),
            ),
            usage_metadata=_DictReplace(),
            model_version=_SimpleStringReplace(),
        )

    def process_chunk(self, chunk: "GenerateContentResponse") -> None:
        self._is_null = False
        values = chunk.model_dump(exclude_unset=True, warnings=False)
        self._values += values

    def _result(self) -> Optional[Dict[str, Any]]:

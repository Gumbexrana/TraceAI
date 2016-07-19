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

from fi_instrumentation import safe_json_dumps
from fi_instrumentation.fi_types import (
    FiMimeTypeValues,
    MessageAttributes,
    SpanAttributes,
    ToolCallAttributes,
)
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue
from traceai_anthropic._utils import (
    _as_output_attributes,
    _finish_tracing,
    _ValueAndType,
)
from traceai_anthropic._with_span import _WithSpan
from wrapt import ObjectProxy

if TYPE_CHECKING:
    from anthropic import Stream
    from anthropic.types import Completion, RawMessageStreamEvent
    from anthropic.types.raw_content_block_delta_event import RawContentBlockDeltaEvent
    from anthropic.types.text_block import TextBlock
    from anthropic.types.tool_use_block import ToolUseBlock


class _Stream(ObjectProxy):  # type: ignore
    __slots__ = (
        "_response_accumulator",
        "_with_span",
        "_is_finished",
    )

    def __init__(
        self,
        stream: "Stream[Completion]",
        with_span: _WithSpan,
    ) -> None:
        super().__init__(stream)
        self._response_accumulator = _ResponseAccumulator()
        self._with_span = with_span

    def __iter__(self) -> Iterator["Completion"]:
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
        status = trace_api.Status(
            status_code=trace_api.StatusCode.OK,
        )
        self._finish_tracing(status=status)

    async def __aiter__(self) -> AsyncIterator["Completion"]:
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
        _finish_tracing(
            with_span=self._with_span,
            has_attributes=_ResponseExtractor(
                response_accumulator=self._response_accumulator
            ),
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
            completion=_StringAccumulator(),
            stop=_SimpleStringReplace(),
            stop_reason=_SimpleStringReplace(),
        )

    def process_chunk(self, chunk: "Completion") -> None:
        self._is_null = False
        values = chunk.model_dump(exclude_unset=True, warnings=False)
        self._values += values

    def _result(self) -> Optional[Dict[str, Any]]:
        if self._is_null:
            return None
        return dict(self._values)


class _ResponseExtractor:
    __slots__ = ("_response_accumulator",)

    def __init__(
        self,
        response_accumulator: _ResponseAccumulator,
    ) -> None:
        self._response_accumulator = response_accumulator

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if not (result := self._response_accumulator._result()):
            return

        json_string = safe_json_dumps(result)
        yield SpanAttributes.OUTPUT_VALUE, json_string

        yield from _as_output_attributes(
            _ValueAndType(json_string, FiMimeTypeValues.JSON)
        )

    def get_extra_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if not (result := self._response_accumulator._result()):
            return
        if completion := result.get("completion", ""):
            yield SpanAttributes.GEN_AI_OUTPUT_MESSAGES, completion


class _MessagesStream(ObjectProxy):  # type: ignore
    __slots__ = (
        "_response_accumulator",
        "_with_span",
        "_is_finished",
    )

    def __init__(
        self,
        stream: "Stream[RawMessageStreamEvent]",
        with_span: _WithSpan,
    ) -> None:
        super().__init__(stream)
        self._response_accumulator = _MessageResponseAccumulator()
        self._with_span = with_span

    def __iter__(self) -> Iterator["RawMessageStreamEvent"]:
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
        status = trace_api.Status(
            status_code=trace_api.StatusCode.OK,
        )
        self._finish_tracing(status=status)

    async def __aiter__(self) -> AsyncIterator["RawMessageStreamEvent"]:
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

import logging
from typing import Any, AsyncIterator, Generator, Iterator, Optional, Protocol, Tuple

from mistralai.models import CompletionEvent
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue
from traceai_mistralai._utils import _finish_tracing
from traceai_mistralai._with_span import _WithSpan
from wrapt import ObjectProxy

__all__ = (
    "_Stream",
    "_ResponseAccumulator",
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _ResponseAccumulator(Protocol):
    def process_chunk(self, chunk: Any) -> None: ...

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]: ...

    def get_extra_attributes(self) -> Iterator[Tuple[str, AttributeValue]]: ...


class _Stream(ObjectProxy):  # type: ignore
    __slots__ = (
        "_self_with_span",
        "_self_iteration_count",
        "_self_is_finished",
        "_self_response_accumulator",
    )

    def __init__(
        self,
        stream: Optional[Generator[CompletionEvent, None, None]],
        with_span: _WithSpan,
        response_accumulator: Optional[_ResponseAccumulator] = None,
    ) -> None:
        super().__init__(stream)
        self._self_with_span = with_span
        self._self_iteration_count = 0
        self._self_is_finished = with_span.is_finished
        self._self_response_accumulator = response_accumulator

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        try:
            chunk = self.__wrapped__.__next__()
        except StopIteration:
            if not self._self_is_finished:
                status = trace_api.Status(status_code=trace_api.StatusCode.OK)
                self._finish_tracing(status=status)
            raise
        except Exception as exception:
            if not self._self_is_finished:
                status = trace_api.Status(
                    status_code=trace_api.StatusCode.ERROR,
                    description=f"{type(exception).__name__}: {exception}",
                )
                self._self_with_span.record_exception(exception)
                self._finish_tracing(status=status)
            raise
        else:
            self._process_chunk(chunk)
            return chunk

    def _process_chunk(self, chunk: Any) -> None:
        if not self._self_iteration_count:

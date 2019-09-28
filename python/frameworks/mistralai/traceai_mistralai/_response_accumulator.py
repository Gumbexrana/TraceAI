from collections import defaultdict
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    Type,
)

from fi_instrumentation import safe_json_dumps
from fi_instrumentation.fi_types import FiMimeTypeValues, SpanAttributes
from opentelemetry.util.types import AttributeValue
from traceai_mistralai._utils import _to_dict

if TYPE_CHECKING:
    from mistralai.models import CompletionEvent

__all__ = ("_ChatCompletionAccumulator",)


class _CanGetAttributesFromResponse(Protocol):
    def get_attributes_from_response(
        self,
        response: Any,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]: ...


class _ChatCompletionAccumulator:
    __slots__ = (
        "_is_null",
        "_values",
        "_cached_result",
        "_request_parameters",
        "_response_attributes_extractor",
        "_chat_completion_type",
        "_content",
        "_raw_data",
    )

    def __init__(
        self,
        request_parameters: Mapping[str, Any],
        chat_completion_type: Type["CompletionEvent"],
        response_attributes_extractor: Optional[_CanGetAttributesFromResponse] = None,
    ) -> None:
        self._chat_completion_type = chat_completion_type
        self._request_parameters = request_parameters
        self._response_attributes_extractor = response_attributes_extractor
        self._is_null = True
        self._cached_result: Optional[Dict[str, Any]] = None
        self._content: List[str] = []
        self._raw_data: List[Dict[str, Any]] = []
        self._values = _ValuesAccumulator(
            data=_ValuesAccumulator(
                choices=_IndexedAccumulator(
                    lambda: _ValuesAccumulator(
                        message=_ValuesAccumulator(
                            content=_StringAccumulator(),
                            tool_calls=_IndexedAccumulator(
                                lambda: _ValuesAccumulator(
                                    function=_ValuesAccumulator(
                                        arguments=_StringAccumulator()
                                    ),
                                )
                            ),
                        ),
                    ),
                ),
            ),
        )

    def process_chunk(self, chunk: Any) -> None:
        raw_chunk = _to_dict(chunk)
        self._raw_data.append(raw_chunk)

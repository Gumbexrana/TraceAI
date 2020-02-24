import warnings
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
from traceai_openai._utils import _as_output_attributes, _to_dict, _ValueAndType, _io_value_and_type

if TYPE_CHECKING:
    from openai.types import Completion
    from openai.types.chat import ChatCompletion
    from openai.types.responses.response import Response
    from openai.types.responses.response_completed_event import ResponseCompletedEvent

__all__ = (
    "_CompletionAccumulator",
    "_ChatCompletionAccumulator",
    "_ResponsesAccumulator",
)


class _CanGetAttributesFromResponse(Protocol):
    def get_attributes_from_response(
        self,
        response: Any,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]: ...


class _ResponsesAccumulator:
    __slots__ = (
        "_is_null",
        "_values",
        "_cached_result",
        "_request_parameters",
        "_response_attributes_extractor",
        "_chat_completion_type",
    )

    def __init__(
        self,
        request_parameters: Mapping[str, Any],
        chat_completion_type: Type["Response"],
        response_attributes_extractor: Optional[_CanGetAttributesFromResponse] = None,
    ) -> None:
        self._chat_completion_type = chat_completion_type
        self._request_parameters = request_parameters
        self._response_attributes_extractor = response_attributes_extractor
        self._is_null = True
        self._cached_result: Optional[Dict[str, Any]] = None
        self._values: Optional["ResponseCompletedEvent"] = None

    def process_chunk(self, chunk: Any) -> None:
        if type(chunk).__name__ == "ResponseCompletedEvent":
            self._is_null = False
            self._cached_result = None
            self._values = chunk

    def _result(self) -> Any:
        if self._is_null:
            return None
        if self._values:
            return self._values.response

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if not (result := self._result()):
            return
        # Extract just the assistant's response content, not the full API response
        output_content = None
        if hasattr(result, "output") and result.output:
            # For Responses API, extract output items
            output_items = []
            for item in result.output:
                if hasattr(item, "content") and item.content:
                    for content in item.content:
                        if hasattr(content, "text"):
                            output_items.append(content.text)
            if output_items:
                output_content = " ".join(output_items)

        if output_content:
            yield SpanAttributes.OUTPUT_VALUE, output_content
        else:
            # Fallback to full response if we can't extract content
            yield from _as_output_attributes(
                _io_value_and_type(result),
            )

    def get_extra_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if not (result := self._result()):
            return
        if self._response_attributes_extractor:
            yield from self._response_attributes_extractor.get_attributes_from_response(
                response=result,
                request_parameters=self._request_parameters,
            )


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
        chat_completion_type: Type["ChatCompletion"],
        response_attributes_extractor: Optional[_CanGetAttributesFromResponse] = None,
    ) -> None:
        self._chat_completion_type = chat_completion_type
        self._request_parameters = request_parameters
        self._response_attributes_extractor = response_attributes_extractor
        self._is_null = True
        self._content: List[str] = []
        self._raw_data: List[Dict[str, Any]] = []
        self._cached_result: Optional[Dict[str, Any]] = None
        self._values = _ValuesAccumulator(
            choices=_IndexedAccumulator(
                lambda: _ValuesAccumulator(
                    message=_ValuesAccumulator(
                        content=_StringAccumulator(),
                        function_call=_ValuesAccumulator(
                            arguments=_StringAccumulator()
                        ),
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
        )

    def process_chunk(self, chunk: Any) -> None:
        raw_chunk = _to_dict(chunk)
        self._raw_data.append(raw_chunk)
        self._is_null = False

        if "choices" in raw_chunk:
            choices = raw_chunk["choices"]
            if choices and isinstance(choices, list):
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content is not None:
                    self._content.append(content)

    def _result(self) -> Optional[Dict[str, Any]]:
        if self._is_null:
            return None
        if not self._cached_result:
            self._cached_result = dict(self._values)
        return self._cached_result

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        # Combine the accumulated content
        output_value = "".join(self._content)

        if self._raw_data:
            last_chunk = self._raw_data[-1]
            if "usage" in last_chunk:
                usage = last_chunk["usage"]
                total_tokens = usage.get("total_tokens")
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")

                if total_tokens is not None:
                    yield SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS, total_tokens
                if prompt_tokens is not None:
                    yield SpanAttributes.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens
                if completion_tokens is not None:
                    yield SpanAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens

        # Only yield the assistant's response content, not the raw API response
        yield SpanAttributes.OUTPUT_VALUE, output_value

    def get_extra_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if not (result := self._result()):
            return
        if self._response_attributes_extractor:
            yield from self._response_attributes_extractor.get_attributes_from_response(
                self._chat_completion_type.construct(**result),
                self._request_parameters,
            )


class _CompletionAccumulator:
    __slots__ = (
        "_is_null",
        "_values",
        "_cached_result",
        "_request_parameters",
        "_response_attributes_extractor",
        "_completion_type",
    )

    def __init__(
        self,
        request_parameters: Mapping[str, Any],
        completion_type: Type["Completion"],
        response_attributes_extractor: Optional[_CanGetAttributesFromResponse] = None,
    ) -> None:
        self._completion_type = completion_type
        self._request_parameters = request_parameters
        self._response_attributes_extractor = response_attributes_extractor
        self._is_null = True
        self._cached_result: Optional[Dict[str, Any]] = None
        self._values = _ValuesAccumulator(
            choices=_IndexedAccumulator(
                lambda: _ValuesAccumulator(text=_StringAccumulator())
            ),
        )

    def process_chunk(self, chunk: "Completion") -> None:
        self._is_null = False
        self._cached_result = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # `warnings=False` in `model_dump()` is only supported in Pydantic v2
            values = chunk.model_dump(exclude_unset=True)
        self._values += values

    def _result(self) -> Optional[Dict[str, Any]]:
        if self._is_null:
            return None
        if not self._cached_result:
            self._cached_result = dict(self._values)
        return self._cached_result

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if not (result := self._result()):
            return
        # Extract just the text content for legacy completions API
        output_content = None
        choices = result.get("choices", [])
        if choices:

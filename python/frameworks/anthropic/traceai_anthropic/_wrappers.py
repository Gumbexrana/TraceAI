import logging
from abc import ABC
from contextlib import contextmanager
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
)

import opentelemetry.context as context_api
from anthropic.types import Message, MessageTokensCount, TextBlock, ToolUseBlock
from fi_instrumentation import get_attributes_from_context, safe_json_dumps
from fi_instrumentation.fi_types import (
    DocumentAttributes,
    EmbeddingAttributes,
    FiLLMProviderValues,
    FiLLMSystemValues,
    FiMimeTypeValues,
    FiSpanKindValues,
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)
from opentelemetry import trace as trace_api
from opentelemetry.trace import INVALID_SPAN
from traceai_anthropic._stream import _MessagesStream, _Stream
from traceai_anthropic._with_span import _WithSpan

if TYPE_CHECKING:
    from anthropic.types import Usage
    from pydantic import BaseModel

logger = logging.getLogger(__name__)


class _WithTracer(ABC):
    """
    Base class for wrappers that need a tracer.
    """

    def __init__(self, tracer: trace_api.Tracer, *args: Any, **kwargs: Any) -> None:

        super().__init__(*args, **kwargs)
        self._tracer = tracer

    @contextmanager
    def _start_as_current_span(
        self, span_name: str, attributes: Optional[Mapping[str, Any]] = None
    ) -> Iterator[_WithSpan]:
        try:
            span = self._tracer.start_span(
                name=span_name,
                record_exception=False,
                set_status_on_exception=False,
                attributes=attributes,
            )
        except Exception as e:
            print(f"Error creating span: {e}")
            span = INVALID_SPAN
        with trace_api.use_span(
            span,
            end_on_exit=False,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            yield _WithSpan(span=span)


class _BaseWrapper(_WithTracer):
    """Base wrapper class with common functionality"""

    def _add_io_to_span_attributes(
        self,
        span: _WithSpan,
        input_data: Optional[Any] = None,
        output_data: Optional[Any] = None,
    ) -> None:
        """Add input/output data to span attributes"""
        # Input/output are already captured in gen_ai.input/output.messages
        # No need for redundant attributes
        pass


class _CompletionsWrapper(_BaseWrapper):
    """
    Wrapper for the pipeline processing
    Captures all calls to the pipeline
    """

    __slots__ = "_response_accumulator"

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        arguments = kwargs
        llm_prompt = dict(arguments).pop("system", None)
        llm_input_messages = dict(arguments).pop("messages", None)
        llm_invocation_parameters = _get_invocation_parameters(arguments)

        extracted_data = _extract_image_data(llm_input_messages)
        llm_filtered_input_messages = extracted_data["filtered_messages"]
        input_images = extracted_data["input_images"]
        eval_input = extracted_data["eval_input"]
        query = extracted_data["query"]

        with self._start_as_current_span(
            span_name="Completions",
            attributes=dict(
                chain(
                    get_attributes_from_context(),
                    _get_llm_model_name_from_input(arguments),
                    _get_llm_provider(),
                    _get_llm_system(),
                    _get_llm_span_kind(),
                    _get_llm_prompts(llm_prompt),
                    _get_inputs(llm_filtered_input_messages),
                    _get_llm_invocation_parameters(llm_invocation_parameters),
                    _get_llm_tools(llm_invocation_parameters),
                    _get_image_inputs(input_images),
                    _get_eval_input(eval_input),
                    _get_raw_input(arguments),
                    _get_query(query),
                )
            ),
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
                span.set_attributes(dict(_get_raw_output(response)))

                # Add I/O to span attributes instead of sending separately
                self._add_io_to_span_attributes(
                    span=span,
                    output_data=(
                        response.completion if hasattr(response, "completion") else None
                    ),
                )

            except Exception as exception:
                print(f"Error in completion: {exception}")
                span.set_status(
                    trace_api.Status(trace_api.StatusCode.ERROR, str(exception))
                )
                span.record_exception(exception)
                raise

            streaming = kwargs.get("stream", False)
            if streaming:
                return _Stream(response, span)
            else:
                span.set_status(trace_api.StatusCode.OK)
                span.set_attributes(dict(_get_outputs(response)))
                span.finish_tracing()
                return response


class _AsyncCompletionsWrapper(_WithTracer):
    """
    Wrapper for the pipeline processing
    Captures all calls to the pipeline
    """

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        arguments = kwargs
        llm_prompt = dict(arguments).pop("system", None)
        llm_input_messages = dict(arguments).pop("messages", None)
        invocation_parameters = _get_invocation_parameters(arguments)

        extracted_data = _extract_image_data(llm_input_messages)
        filtered_llm_input_messages = extracted_data["filtered_messages"]
        input_images = extracted_data["input_images"]
        eval_input = extracted_data["eval_input"]
        query = extracted_data["query"]

        with self._start_as_current_span(
            span_name="AsyncCompletions",
            attributes=dict(
                chain(
                    get_attributes_from_context(),
                    _get_llm_model_name_from_input(arguments),
                    _get_llm_provider(),
                    _get_llm_system(),
                    _get_llm_span_kind(),
                    _get_llm_prompts(llm_prompt),
                    _get_inputs(filtered_llm_input_messages),
                    _get_llm_invocation_parameters(invocation_parameters),
                    _get_llm_tools(invocation_parameters),
                    _get_image_inputs(input_images),
                    _get_eval_input(eval_input),
                    _get_raw_input(arguments),
                    _get_query(query),
                )
            ),
        ) as span:
            try:
                response = await wrapped(*args, **kwargs)
                span.set_attributes(dict(_get_raw_output(response)))
            except Exception as exception:
                span.set_status(
                    trace_api.Status(trace_api.StatusCode.ERROR, str(exception))
                )
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
            streaming = kwargs.get("stream", False)
            if streaming:
                return _Stream(response, span)
            else:
                span.set_status(trace_api.StatusCode.OK)
                span.set_attributes(dict(_get_async_outputs(response)))
                span.finish_tracing()
                return response


class _MessagesWrapper(_BaseWrapper):
    """
    Wrapper for the pipeline processing
    Captures all calls to the pipeline
    """

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        arguments = kwargs
        llm_input_messages = dict(arguments).pop("messages", None)
        llm_prompt = dict(arguments).pop("system", None)
        invocation_parameters = _get_invocation_parameters(arguments)

        extracted_data = _extract_image_data(llm_input_messages)
        llm_filtered_input_messages = extracted_data["filtered_messages"]
        input_images = extracted_data["input_images"]
        eval_input = extracted_data["eval_input"]
        query = extracted_data["query"]

        with self._start_as_current_span(
            span_name="Messages",
            attributes=dict(
                chain(
                    get_attributes_from_context(),
                    _get_llm_model_name_from_input(arguments),
                    _get_llm_provider(),
                    _get_llm_system(),
                    _get_llm_span_kind(),
                    _get_llm_prompts(llm_prompt),
                    _get_llm_input_messages(llm_input_messages),
                    _get_llm_invocation_parameters(invocation_parameters),
                    _get_llm_tools(invocation_parameters),
                    _get_inputs(llm_filtered_input_messages),
                    _get_image_inputs(input_images),
                    _get_eval_input(eval_input),
                    _get_raw_input(arguments),
                    _get_query(query),
                )
            ),
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
                streaming = kwargs.get("stream", False)
                if not streaming:
                    span.set_attributes(dict(_get_raw_output(response)))

                # Get output content
                output_content = None
                if hasattr(response, "content"):
                    if isinstance(response.content, list):
                        for block in response.content:
                            if hasattr(block, "text"):
                                output_content = block.text
                                break
                            elif isinstance(block, dict) and "text" in block:
                                output_content = block["text"]
                                break

                # Add I/O to span attributes instead of sending separately
                self._add_io_to_span_attributes(span=span, output_data=output_content)

            except Exception as exception:
                span.set_status(
                    trace_api.Status(trace_api.StatusCode.ERROR, str(exception))
                )
                span.record_exception(exception)
                raise

            streaming = kwargs.get("stream", False)
            if streaming:
                return _MessagesStream(response, span)
            else:
                span.set_status(trace_api.StatusCode.OK)
                span.set_attributes(
                    dict(
                        chain(
                            _get_llm_model_name_from_response(response),
                            _get_output_messages(response),
                            _get_llm_token_counts(response.usage),
                            _get_outputs(response),
                        )
                    )
                )
                span.finish_tracing()
                return response


class _AsyncMessagesWrapper(_WithTracer):
    """
    Wrapper for the pipeline processing
    Captures all calls to the pipeline
    """

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

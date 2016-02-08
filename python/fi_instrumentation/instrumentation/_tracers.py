import asyncio
import collections
import inspect
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import datetime
from secrets import randbits
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,  # type: ignore[attr-defined]
    Awaitable,
    Callable,
    Dict,
    Iterator,
    Literal,
    Optional,
    Tuple,
    Union,
    _TypedDictMeta,
    cast,
    get_args,
    get_origin,
)

import wrapt  # type: ignore[import-untyped]
from fi_instrumentation.fi_types import FiSpanKindValues, SpanAttributes
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY, Context, get_value
from opentelemetry.sdk.trace.id_generator import IdGenerator, RandomIdGenerator
from opentelemetry.trace import (
    INVALID_SPAN,
    INVALID_SPAN_ID,
    INVALID_TRACE_ID,
    Link,
    Span,
    SpanKind,
    Status,
    StatusCode,
    Tracer,
    use_span,
)
from opentelemetry.util.types import Attributes
from typing_extensions import ParamSpec, TypeVar, _AnnotatedAlias, overload

from ._attributes import (
    get_input_attributes,
    get_span_kind_attributes,
    get_tool_attributes,
)
from ._spans import FiSpan
from .config import TraceConfig
from .context_attributes import get_attributes_from_context

if TYPE_CHECKING:
    from ._types import FiSpanKind

ParametersType = ParamSpec("ParametersType")
ReturnType = TypeVar("ReturnType")

pydantic: Optional[ModuleType]
try:
    import pydantic  # try to import without adding a dependency
except ImportError:
    pydantic = None


class _IdGenerator(IdGenerator):
    """
    An IdGenerator that uses a different source of randomness to
    avoid being affected by seeds set by user application.
    """

    def generate_span_id(self) -> int:
        while (span_id := randbits(64)) == INVALID_SPAN_ID:
            continue
        return span_id

    def generate_trace_id(self) -> int:
        while (trace_id := randbits(128)) == INVALID_TRACE_ID:
            continue
        return trace_id


class FITracer(wrapt.ObjectProxy):  # type: ignore[misc]
    def __init__(self, wrapped: Tracer, config: TraceConfig = TraceConfig()) -> None:
        super().__init__(wrapped)
        self._self_config = config
        self._self_id_generator = _IdGenerator()

    @property
    def id_generator(self) -> IdGenerator:
        ans = getattr(self.__wrapped__, "id_generator", None)
        if ans and ans.__class__ is RandomIdGenerator:
            return self._self_id_generator
        return cast(IdGenerator, ans)

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        context: Optional[Context] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Attributes = None,
        links: Optional["Sequence[Link]"] = (),
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
        *,
        fi_span_kind: Optional["FiSpanKind"] = None,
    ) -> Iterator[FiSpan]:
        span = self.start_span(
            name=name,
            fi_span_kind=fi_span_kind,
            context=context,
            kind=kind,
            attributes=attributes,
            links=links,
            start_time=start_time,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
        )
        with use_span(
            span,  # type: ignore[arg-type]
            end_on_exit=end_on_exit,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
        ) as current_span:
            yield cast(FiSpan, current_span)

    def start_span(
        self,
        name: str,
        context: Optional[Context] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Attributes = None,
        links: Optional["Sequence[Link]"] = (),
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        *,
        fi_span_kind: Optional["FiSpanKind"] = None,
    ) -> FiSpan:
        otel_span: Span
        if get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            otel_span = INVALID_SPAN
        else:
            tracer = cast(Tracer, self.__wrapped__)
            otel_span = tracer.__class__.start_span(
                self,
                name=name,
                context=context,
                kind=kind,
                attributes=None,
                links=links,
                start_time=start_time,
                record_exception=record_exception,
                set_status_on_exception=set_status_on_exception,
            )
        fi_span = FiSpan(otel_span, config=self._self_config)
        if attributes:
            fi_span.set_attributes(dict(attributes))
        if fi_span_kind is not None:
            fi_span.set_attributes(get_span_kind_attributes(fi_span_kind))
        fi_span.set_attributes(dict(get_attributes_from_context()))
        return fi_span

    @overload  # for @tracer.agent usage (no parameters)
    def agent(
        self,
        wrapped_function: Callable[ParametersType, ReturnType],
        /,
        *,
        name: None = None,
    ) -> Callable[ParametersType, ReturnType]: ...

    @overload  # for @tracer.agent(name="name") usage (with parameters)
    def agent(
        self,
        wrapped_function: None = None,
        /,
        *,
        name: Optional[str] = None,
    ) -> Callable[
        [Callable[ParametersType, ReturnType]], Callable[ParametersType, ReturnType]
    ]: ...

    def agent(
        self,
        wrapped_function: Optional[Callable[ParametersType, ReturnType]] = None,
        /,
        *,
        name: Optional[str] = None,
    ) -> Union[
        Callable[ParametersType, ReturnType],
        Callable[
            [Callable[ParametersType, ReturnType]], Callable[ParametersType, ReturnType]
        ],
    ]:
        return self._chain(
            wrapped_function,
            kind=FiSpanKindValues.AGENT,  # chains and agents differ only in span kind
            name=name,
        )

    @overload  # for @tracer.chain usage (no parameters)
    def chain(
        self,
        wrapped_function: Callable[ParametersType, ReturnType],
        /,
        *,
        name: None = None,
    ) -> Callable[ParametersType, ReturnType]: ...

    @overload  # for @tracer.chain(name="name") usage (with parameters)
    def chain(
        self,
        wrapped_function: None = None,
        /,
        *,
        name: Optional[str] = None,
    ) -> Callable[
        [Callable[ParametersType, ReturnType]], Callable[ParametersType, ReturnType]
    ]: ...

    def chain(
        self,
        wrapped_function: Optional[Callable[ParametersType, ReturnType]] = None,
        /,
        *,
        name: Optional[str] = None,
    ) -> Union[
        Callable[ParametersType, ReturnType],
        Callable[
            [Callable[ParametersType, ReturnType]], Callable[ParametersType, ReturnType]
        ],
    ]:
        return self._chain(wrapped_function, kind=FiSpanKindValues.CHAIN, name=name)

    def _chain(
        self,
        wrapped_function: Optional[Callable[ParametersType, ReturnType]] = None,
        /,
        *,
        kind: FiSpanKindValues,
        name: Optional[str] = None,
    ) -> Union[
        Callable[ParametersType, ReturnType],
        Callable[
            [Callable[ParametersType, ReturnType]], Callable[ParametersType, ReturnType]
        ],
    ]:
        @wrapt.decorator  # type: ignore[misc]
        def sync_wrapper(
            wrapped: Callable[ParametersType, ReturnType],
            instance: Any,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> ReturnType:
            tracer = self
            with _chain_context(
                tracer=tracer,
                name=name,
                kind=kind,
                wrapped=wrapped,
                instance=instance,
                args=args,
                kwargs=kwargs,
            ) as chain_context:
                output = wrapped(*args, **kwargs)
                return chain_context.process_output(output)

        @wrapt.decorator  #  type: ignore[misc]
        async def async_wrapper(
            wrapped: Callable[ParametersType, Awaitable[ReturnType]],
            instance: Any,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> ReturnType:
            tracer = self
            with _chain_context(
                tracer=tracer,
                name=name,
                kind=kind,
                wrapped=wrapped,
                instance=instance,
                args=args,
                kwargs=kwargs,
            ) as chain_context:
                output = await wrapped(*args, **kwargs)
                return chain_context.process_output(output)

        if wrapped_function is not None:
            if asyncio.iscoroutinefunction(wrapped_function):
                return async_wrapper(wrapped_function)  # type: ignore[no-any-return]
            return sync_wrapper(wrapped_function)  # type: ignore[no-any-return]
        return lambda f: (
            async_wrapper(f) if asyncio.iscoroutinefunction(f) else sync_wrapper(f)
        )

    @overload  # for @tracer.tool usage (no parameters)
    def tool(
        self,
        wrapped_function: Callable[ParametersType, ReturnType],
        /,
        *,
        name: None = None,
        description: Optional[str] = None,
        parameters: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Callable[ParametersType, ReturnType]: ...

    @overload  # for @tracer.tool(name="name") usage (with parameters)
    def tool(
        self,
        wrapped_function: None = None,
        /,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Callable[
        [Callable[ParametersType, ReturnType]], Callable[ParametersType, ReturnType]
    ]: ...

    def tool(
        self,
        wrapped_function: Optional[Callable[ParametersType, ReturnType]] = None,
        /,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Union[
        Callable[ParametersType, ReturnType],
        Callable[
            [Callable[ParametersType, ReturnType]], Callable[ParametersType, ReturnType]
        ],
    ]:
        @wrapt.decorator  # type: ignore[misc]
        def sync_wrapper(
            wrapped: Callable[ParametersType, ReturnType],
            instance: Any,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> ReturnType:
            tracer = self
            with _tool_context(
                tracer=tracer,
                name=name,
                description=description,
                parameters=parameters,
                wrapped=wrapped,
                instance=instance,
                args=args,
                kwargs=kwargs,
            ) as tool_context:
                output = wrapped(*args, **kwargs)
                return tool_context.process_output(output)

        @wrapt.decorator  #  type: ignore[misc]
        async def async_wrapper(
            wrapped: Callable[ParametersType, Awaitable[ReturnType]],
            instance: Any,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> ReturnType:
            tracer = self
            with _tool_context(
                tracer=tracer,
                name=name,
                description=description,
                parameters=parameters,
                wrapped=wrapped,
                instance=instance,
                args=args,
                kwargs=kwargs,
            ) as tool_context:
                output = await wrapped(*args, **kwargs)
                return tool_context.process_output(output)

        if wrapped_function is not None:
            if asyncio.iscoroutinefunction(wrapped_function):
                return async_wrapper(wrapped_function)  # type: ignore[no-any-return]
            return sync_wrapper(wrapped_function)  # type: ignore[no-any-return]
        return lambda f: (
            async_wrapper(f) if asyncio.iscoroutinefunction(f) else sync_wrapper(f)
        )


class _ChainContext:
    def __init__(self, span: "FiSpan") -> None:
        self._span = span

    def process_output(self, output: ReturnType) -> ReturnType:
        attributes = getattr(self._span, "attributes", {})
        has_output = OUTPUT_VALUE in attributes
        if not has_output:
            self._span.set_output(value=output)
        return output


@contextmanager
def _chain_context(
    *,
    tracer: "FITracer",
    name: Optional[str],
    kind: FiSpanKindValues,
    wrapped: Callable[ParametersType, ReturnType],
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Iterator[_ChainContext]:
    span_name = name or _infer_span_name(instance=instance, callable=wrapped)
    bound_args = inspect.signature(wrapped).bind(*args, **kwargs)
    bound_args.apply_defaults()
    arguments = bound_args.arguments

    if len(arguments) == 1:
        argument = next(iter(arguments.values()))
        input_attributes = get_input_attributes(argument)
    else:
        input_attributes = get_input_attributes(arguments)

    with tracer.start_as_current_span(
        span_name,
        fi_span_kind=kind,
        attributes=input_attributes,
    ) as span:
        context = _ChainContext(span=span)
        yield context


class _ToolContext:
    def __init__(self, span: "FiSpan") -> None:
        self._span = span

    def process_output(self, output: ReturnType) -> ReturnType:
        attributes = getattr(self._span, "attributes", {})
        has_output = OUTPUT_VALUE in attributes
        if not has_output:
            self._span.set_output(value=output)
        return output


@contextmanager
def _tool_context(
    *,
    tracer: "FITracer",
    name: Optional[str],
    description: Optional[str],
    parameters: Optional[Union[str, Dict[str, Any]]],
    wrapped: Callable[ParametersType, ReturnType],

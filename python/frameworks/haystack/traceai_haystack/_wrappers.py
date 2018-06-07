from enum import Enum, auto
from inspect import BoundArguments, Parameter, signature
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    get_type_hints,
)

import opentelemetry.context as context_api
from fi_instrumentation import get_attributes_from_context, safe_json_dumps
from fi_instrumentation.fi_types import (
    DocumentAttributes,
    EmbeddingAttributes,
    FiMimeTypeValues,
    FiSpanKindValues,
    MessageAttributes,
    RerankerAttributes,
    SpanAttributes,
    ToolCallAttributes,
)
from opentelemetry import trace as trace_api
from typing_extensions import TypeGuard, assert_never

if TYPE_CHECKING:
    from haystack import Document, Pipeline
    from haystack.core.component import Component


class _PipelineRunComponentWrapper:
    """
    Components in Haystack are defined as a duck-typed protocol with a `run`
    method invoked by `haystack.Pipeline._run_component`. We dynamically wrap
    the component `run` method if it is not already wrapped from within
    `haystack.Pipeline._run_component`.
    """

    def __init__(
        self,
        tracer: trace_api.Tracer,
        wrap_component_run_method: Callable[[type[Any], Callable[..., Any]], None],
    ) -> None:
        self._tracer = tracer
        self._wrap_component_run_method = wrap_component_run_method

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        arguments = _get_bound_arguments(wrapped, *args, **kwargs).arguments
        if (component := arguments.get("component")) is not None and (
            component_instance := component.get("instance")
        ) is not None:
            component_cls = component_instance.__class__
            self._wrap_component_run_method(component_cls, component_instance.run)
        return wrapped(*args, **kwargs)


class _ComponentRunWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        component = instance
        component_class_name = _get_component_class_name(component)
        bound_arguments = _get_bound_arguments(wrapped, *args, **kwargs)
        arguments = bound_arguments.arguments

        with self._tracer.start_as_current_span(
            name=_get_component_span_name(component_class_name)
        ) as span:
            span.set_attributes(
                {
                    **dict(get_attributes_from_context()),
                    **dict(_get_input_attributes(arguments)),
                }
            )
            if (
                component_type := _get_component_type(component)
            ) is ComponentType.GENERATOR:
                span.set_attributes(
                    {
                        **dict(_get_span_kind_attributes(LLM)),
                        **dict(_get_llm_input_message_attributes(arguments)),
                    }
                )
            elif component_type is ComponentType.EMBEDDER:
                span.set_attributes(
                    {
                        **dict(_get_span_kind_attributes(EMBEDDING)),
                        **dict(_get_embedding_model_attributes(component)),
                    }
                )
            elif component_type is ComponentType.RANKER:
                span.set_attributes(
                    {
                        **dict(_get_span_kind_attributes(RERANKER)),
                        **dict(_get_reranker_model_attributes(component)),
                        **dict(_get_reranker_request_attributes(arguments)),
                    }
                )
            elif component_type is ComponentType.RETRIEVER:
                span.set_attributes(dict(_get_span_kind_attributes(RETRIEVER)))
            elif component_type is ComponentType.PROMPT_BUILDER:
                span.set_attributes(
                    {
                        **dict(_get_span_kind_attributes(LLM)),
                        **dict(
                            _get_llm_prompt_template_attributes_from_prompt_builder(
                                component, bound_arguments
                            )
                        ),
                    }
                )
            elif component_type is ComponentType.UNKNOWN:
                span.set_attributes(dict(_get_span_kind_attributes(CHAIN)))
            else:
                assert_never(component_type)

            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(
                    trace_api.Status(trace_api.StatusCode.ERROR, str(exception))
                )
                raise
            span.set_attributes(
                dict(_get_component_output_attributes(response, component_type))
            )
            span.set_status(trace_api.StatusCode.OK)
            if component_type is ComponentType.GENERATOR:
                span.set_attributes(
                    {
                        **dict(_get_llm_model_attributes(response)),
                        **dict(_get_llm_output_message_attributes(response)),
                        **dict(_get_llm_token_count_attributes(response)),
                    }
                )
            elif component_type is ComponentType.EMBEDDER:
                span.set_attributes(
                    dict(_get_embedding_attributes(arguments, response))
                )
            elif component_type is ComponentType.RANKER:
                span.set_attributes(dict(_get_reranker_response_attributes(response)))
            elif component_type is ComponentType.RETRIEVER:
                span.set_attributes(dict(_get_retriever_response_attributes(response)))
            elif component_type is ComponentType.PROMPT_BUILDER:
                pass
            elif component_type is ComponentType.UNKNOWN:
                pass
            else:
                assert_never(component_type)

        return response


class _PipelineWrapper:
    """
    Wrapper for the pipeline processing
    Captures all calls to the pipeline
    """

    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        arguments = _get_bound_arguments(wrapped, *args, **kwargs).arguments

        span_name = "Pipeline.run"
        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                **dict(get_attributes_from_context()),
                **dict(_get_span_kind_attributes(CHAIN)),
                **dict(_get_input_attributes(arguments)),
            },
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(
                    trace_api.Status(trace_api.StatusCode.ERROR, str(exception))
                )
                raise
            span.set_attributes(dict(_get_output_attributes(response)))
            span.set_status(trace_api.StatusCode.OK)

        return response


class ComponentType(Enum):
    GENERATOR = auto()
    EMBEDDER = auto()
    RANKER = auto()
    RETRIEVER = auto()
    PROMPT_BUILDER = auto()
    UNKNOWN = auto()


def _get_component_by_name(
    pipeline: "Pipeline", component_name: str
) -> Optional["Component"]:
    """
    Gets the component invoked by `haystack.Pipeline._run_component` (if one exists).
    """
    if (node := pipeline.graph.nodes.get(component_name)) is None or (
        component := node.get("instance")
    ) is None:
        return None
    return component


def _get_component_class_name(component: "Component") -> str:
    """
    Gets the name of the component.
    """
    return str(component.__class__.__name__)


def _get_component_span_name(component_class_name: str) -> str:
    """
    Gets the name of the span for a component.
    """
    return f"{component_class_name}.run"


def _get_component_type(component: "Component") -> ComponentType:
    """
    Haystack has a single `Component` interface that produces unstructured
    outputs. In the absence of typing information, we make a best-effort attempt
    to infer the component type.
    """

    from haystack.components.builders import PromptBuilder

    component_name = _get_component_class_name(component)
    if (run_method := _get_component_run_method(component)) is None:
        return ComponentType.UNKNOWN
    if "Generator" in component_name or _has_generator_output_type(run_method):
        return ComponentType.GENERATOR
    elif "Embedder" in component_name:
        return ComponentType.EMBEDDER
    elif "Ranker" in component_name and _has_ranker_io_types(run_method):
        return ComponentType.RANKER
    elif (
        "Retriever" in component_name or "WebSearch" in component_name
    ) and _has_retriever_io_types(run_method):
        return ComponentType.RETRIEVER
    elif isinstance(component, PromptBuilder):
        return ComponentType.PROMPT_BUILDER
    return ComponentType.UNKNOWN


def _get_component_run_method(component: "Component") -> Optional[Callable[..., Any]]:
    """
    Gets the `run` method for a component (if one exists).
    """
    if callable(run_method := getattr(component, "run", None)):
        return cast(Callable[..., Any], run_method)
    return None


def _get_run_method_output_types(
    run_method: Callable[..., Any]
) -> Optional[Dict[str, type]]:
    """
    Haystack components are decorated with an `output_type` decorator that is
    useful for inferring the component type.

    https://github.com/deepset-ai/haystack/blob/21c507331c98c76aed88cd8046373dfa2a3590e7/haystack/core/component/component.py#L398
    """

    if isinstance(
        (output_types_cache := getattr(run_method, "_output_types_cache", None)), dict
    ):
        return {key: value.type for key, value in output_types_cache.items()}
    return None


def _get_run_method_input_types(
    run_method: Callable[..., Any]
) -> Optional[Dict[str, type]]:
    """
    Gets input types of parameters to the `run` method.
    """
    return get_type_hints(run_method)


def _has_generator_output_type(run_method: Callable[..., Any]) -> bool:
    """

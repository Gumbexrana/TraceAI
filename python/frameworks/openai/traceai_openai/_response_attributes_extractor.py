from __future__ import annotations

import base64
import logging
from importlib import import_module

from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
)

from fi_instrumentation.fi_types import (
    EmbeddingAttributes,
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
    ToolCallAttributes,
)
from opentelemetry.util.types import AttributeValue
from traceai_openai._attributes._responses_api import _ResponsesApiAttributes
from traceai_openai._utils import _get_openai_version, _get_texts

if TYPE_CHECKING:
    from openai.types import Completion, CreateEmbeddingResponse
    from openai.types.chat import ChatCompletion
    from openai.types.images_response import ImagesResponse
    from openai.types.responses.response import Response

__all__ = ("_ResponseAttributesExtractor",)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


try:
    _NUMPY: Optional[ModuleType] = import_module("numpy")
except ImportError:
    _NUMPY = None


class _ResponseAttributesExtractor:
    __slots__ = (
        "_openai",
        "_chat_completion_type",
        "_completion_type",
        "_create_embedding_response_type",
        "_images_response_type",
        "_responses_type",
    )

    def __init__(self, openai: ModuleType) -> None:
        self._openai = openai
        self._chat_completion_type: Type["ChatCompletion"] = (
            openai.types.chat.ChatCompletion
        )
        self._completion_type: Type["Completion"] = openai.types.Completion
        self._responses_type: Type["Response"] = openai.types.responses.response.Response
        self._create_embedding_response_type: Type["CreateEmbeddingResponse"] = (
            openai.types.CreateEmbeddingResponse
        )
        self._images_response_type: Type["ImagesResponse"] = (
            openai.types.images_response.ImagesResponse
        )

    def get_attributes_from_response(
        self,
        response: Any,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if isinstance(response, self._chat_completion_type):
            yield from self._get_attributes_from_chat_completion(
                completion=response,
                request_parameters=request_parameters,
            )
        elif isinstance(response, self._responses_type):
            yield from self._get_attributes_from_responses_response(
                response=response,
                request_parameters=request_parameters,
            )
        elif isinstance(response, self._create_embedding_response_type):
            yield from self._get_attributes_from_create_embedding_response(
                response=response,
                request_parameters=request_parameters,
            )
        elif isinstance(response, self._completion_type):
            yield from self._get_attributes_from_completion(
                completion=response,
                request_parameters=request_parameters,
            )
        elif isinstance(response, self._images_response_type):
            yield from self._get_attributes_from_image_generation(
                data=response.data,
                request_parameters=request_parameters,
            )
        else:
            yield from ()

    def _get_attributes_from_responses_response(
        self,
        response: Response,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield from _ResponsesApiAttributes._get_attributes_from_response(response)

    def _get_attributes_from_chat_completion(
        self,
        completion: "ChatCompletion",
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        # See https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/types/chat/chat_completion.py#L40  # noqa: E501
        if model := getattr(completion, "model", None):
            yield SpanAttributes.GEN_AI_REQUEST_MODEL, model
        if usage := getattr(completion, "usage", None):
            yield from self._get_attributes_from_completion_usage(usage)

        if (choices := getattr(completion, "choices", None)) and isinstance(
            choices, Iterable
        ):
            for choice in choices:
                if (index := getattr(choice, "index", None)) is None:
                    continue
                if message := getattr(choice, "message", None):
                    for key, value in self._get_attributes_from_chat_completion_message(
                        message
                    ):
                        yield f"{SpanAttributes.GEN_AI_OUTPUT_MESSAGES}.{index}.{key}", value

    def _get_attributes_from_completion(
        self,
        completion: "Completion",
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        # See https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/types/completion.py#L13  # noqa: E501
        if model := getattr(completion, "model", None):
            yield SpanAttributes.GEN_AI_REQUEST_MODEL, model
        if usage := getattr(completion, "usage", None):
            yield from self._get_attributes_from_completion_usage(usage)
        if model_prompt := request_parameters.get("prompt"):
            # FIXME: this step should move to request attributes extractor if decoding is not necessary.# noqa: E501
            # prompt: Required[Union[str, List[str], List[int], List[List[int]], None]]
            # See https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/types/completion_create_params.py#L38
            # FIXME: tokens (List[int], List[List[int]]) can't be decoded reliably because model
            # names are not reliable (across OpenAI and Azure).
            if prompts := list(_get_texts(model_prompt, model)):
                yield SpanAttributes.GEN_AI_PROMPTS, prompts

    def _get_attributes_from_create_embedding_response(
        self,
        response: "CreateEmbeddingResponse",
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        # See https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/types/create_embedding_response.py#L20  # noqa: E501
        if usage := getattr(response, "usage", None):
            yield from self._get_attributes_from_embedding_usage(usage)
        # See https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/types/embedding_create_params.py#L23  # noqa: E501
        if model := getattr(response, "model"):
            yield f"{SpanAttributes.EMBEDDING_MODEL_NAME}", model
        if (data := getattr(response, "data", None)) and isinstance(data, Iterable):
            for embedding in data:
                if (index := getattr(embedding, "index", None)) is None:
                    continue
                for key, value in self._get_attributes_from_embedding(embedding):
                    yield f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.{index}.{key}", value

        embedding_input = request_parameters.get("input")
        for index, text in enumerate(_get_texts(embedding_input, model)):
            # FIXME: this step should move to request attributes extractor if decoding is not necessary.# noqa: E501
            # input: Required[Union[str, List[str], List[int], List[List[int]]]]
            # See https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/types/embedding_create_params.py#L12
            # FIXME: tokens (List[int], List[List[int]]) can't be decoded reliably because model
            # names are not reliable (across OpenAI and Azure).
            yield (
                (
                    f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.{index}."
                    f"{EmbeddingAttributes.EMBEDDING_TEXT}"
                ),
                text,
            )

    def _get_attributes_from_embedding(
        self,
        embedding: object,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        # openai.types.Embedding
        # See https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/types/embedding.py#L11  # noqa: E501
        if not (_vector := getattr(embedding, "embedding", None)):
            return
        if (
            isinstance(_vector, Sequence)
            and len(_vector)
            and isinstance(_vector[0], float)
        ):
            vector = list(_vector)
            yield f"{EmbeddingAttributes.EMBEDDING_VECTOR}", vector
        elif isinstance(_vector, str) and _vector and _NUMPY:
            # FIXME: this step should be removed if decoding is not necessary.
            try:
                # See https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/resources/embeddings.py#L100  # noqa: E501
                vector = _NUMPY.frombuffer(
                    base64.b64decode(_vector), dtype="float32"
                ).tolist()
            except Exception:
                logger.exception("Failed to decode embedding")
                pass
            else:
                yield f"{EmbeddingAttributes.EMBEDDING_VECTOR}", vector

    def _get_attributes_from_chat_completion_message(
        self,

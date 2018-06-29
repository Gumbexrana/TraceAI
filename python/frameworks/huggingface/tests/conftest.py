"""Pytest configuration and fixtures for Ollama instrumentation tests."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest


# Create mock modules before any imports
def setup_mocks():
    """Set up all required module mocks."""

    # Helper to create a module-like object
    def create_mock_module(name):
        mod = ModuleType(name)
        return mod

    # Create a proper SpanAttributes class that returns string values
    class MockSpanAttributes:
        GEN_AI_PROVIDER_NAME = "llm.system"
        GEN_AI_PROVIDER_NAME = "llm.provider"
        GEN_AI_REQUEST_MODEL = "llm.model"
        GEN_AI_REQUEST_MODEL = "llm.model"
        LLM_RESPONSE_MODEL = "llm.response_model"
        GEN_AI_REQUEST_PARAMETERS = "llm.invocation_parameters"
        GEN_AI_USAGE_INPUT_TOKENS = "llm.token_count.prompt"
        GEN_AI_USAGE_OUTPUT_TOKENS = "llm.token_count.completion"
        GEN_AI_USAGE_TOTAL_TOKENS = "llm.token_count.total"
        GEN_AI_SPAN_KIND = "fi.span.kind"
        INPUT_VALUE = "input.value"
        OUTPUT_VALUE = "output.value"
        EMBEDDING_MODEL_NAME = "embedding.model"
        LLM_SYSTEM_PROMPT = "llm.system_prompt"
        INPUT_MIME_TYPE = "input.mime_type"
        OUTPUT_MIME_TYPE = "output.mime_type"
        GEN_AI_INPUT_MESSAGES = "llm.input_messages"
        GEN_AI_OUTPUT_MESSAGES = "llm.output_messages"
        EMBEDDING_EMBEDDINGS = "embedding.embeddings"
        OUTPUT_VALUE = "raw.output"
        RERANKER_MODEL_NAME = "reranker.model"
        RERANKER_QUERY = "reranker.query"
        RERANKER_TOP_K = "reranker.top_k"

    class MockMessageAttributes:
        MESSAGE_ROLE = "message.role"
        MESSAGE_CONTENT = "message.content"

    class MockEmbeddingAttributes:
        EMBEDDING_VECTOR = "embedding.vector"

    # Create mock for fi_instrumentation.fi_types
    mock_fi_types = MagicMock()
    mock_fi_types.SpanAttributes = MockSpanAttributes

    # Create proper enum-like objects for FiSpanKindValues
    class MockEnum:
        def __init__(self, value):
            self.value = value

    class MockFiSpanKindValues:
        LLM = MockEnum("LLM")
        EMBEDDING = MockEnum("EMBEDDING")
        RERANKER = MockEnum("RERANKER")

    class MockFiMimeTypeValues:
        JSON = MockEnum("application/json")
        TEXT = MockEnum("text/plain")

    mock_fi_types.FiSpanKindValues = MockFiSpanKindValues
    mock_fi_types.FiMimeTypeValues = MockFiMimeTypeValues

    mock_fi_types.MessageAttributes = MockMessageAttributes
    mock_fi_types.EmbeddingAttributes = MockEmbeddingAttributes

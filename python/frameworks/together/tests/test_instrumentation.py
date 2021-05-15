"""Tests for Together AI instrumentation."""
import json
from unittest.mock import MagicMock, patch

import pytest

from traceai_together import TogetherInstrumentor


class TestTogetherInstrumentor:
    """Tests for TogetherInstrumentor."""

    def test_instrumentation_dependencies(self):
        """Test instrumentation dependencies."""
        instrumentor = TogetherInstrumentor()
        deps = instrumentor.instrumentation_dependencies()
        assert "together >= 1.0.0" in deps


class TestChatCompletionsRequestAttributesExtractor:
    """Tests for chat completions request attributes extraction."""

    def test_extract_chat_attributes(self):
        """Test extracting chat request attributes."""
        from traceai_together._request_attributes_extractor import _ChatCompletionsRequestAttributesExtractor

        extractor = _ChatCompletionsRequestAttributesExtractor()
        request_params = {
            "model": "meta-llama/Llama-3-8b-chat-hf",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
            "temperature": 0.7,
        }

        attributes = dict(extractor.get_attributes_from_request(request_params))

        assert attributes["fi.span.kind"] == "LLM"
        assert attributes["llm.system"] == "together"
        assert attributes["llm.model"] == "meta-llama/Llama-3-8b-chat-hf"

    def test_extract_chat_with_system_message(self):
        """Test extracting chat with system message."""
        from traceai_together._request_attributes_extractor import _ChatCompletionsRequestAttributesExtractor

        extractor = _ChatCompletionsRequestAttributesExtractor()
        request_params = {
            "model": "meta-llama/Llama-3-8b-chat-hf",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }

        extra_attrs = dict(extractor.get_extra_attributes_from_request(request_params))

        # Should have messages
        assert any("input_messages" in k for k in extra_attrs.keys())

    def test_extract_tools(self):
        """Test extracting tools from request."""
        from traceai_together._request_attributes_extractor import _ChatCompletionsRequestAttributesExtractor

        extractor = _ChatCompletionsRequestAttributesExtractor()
        request_params = {
            "model": "meta-llama/Llama-3-8b-chat-hf",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather info",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        }

        extra_attrs = dict(extractor.get_extra_attributes_from_request(request_params))

        # Tools are stored under llm.tools prefix
        tool_keys = [k for k in extra_attrs.keys() if "llm.tools" in k]
        assert len(tool_keys) >= 1


class TestCompletionsRequestAttributesExtractor:
    """Tests for completions request attributes extraction."""

    def test_extract_completions_attributes(self):
        """Test extracting completions request attributes."""
        from traceai_together._request_attributes_extractor import _CompletionsRequestAttributesExtractor

        extractor = _CompletionsRequestAttributesExtractor()
        request_params = {
            "model": "meta-llama/Llama-3-8b-hf",
            "prompt": "The capital of France is",
            "max_tokens": 100,
        }

        attributes = dict(extractor.get_attributes_from_request(request_params))

        assert attributes["fi.span.kind"] == "LLM"
        assert attributes["llm.system"] == "together"
        assert attributes["llm.model"] == "meta-llama/Llama-3-8b-hf"


class TestEmbeddingsRequestAttributesExtractor:
    """Tests for embeddings request attributes extraction."""

    def test_extract_embeddings_attributes(self):
        """Test extracting embeddings request attributes."""
        from traceai_together._request_attributes_extractor import _EmbeddingsRequestAttributesExtractor

        extractor = _EmbeddingsRequestAttributesExtractor()
        request_params = {
            "model": "togethercomputer/m2-bert-80M-8k-retrieval",
            "input": ["Hello", "World"],
        }

        attributes = dict(extractor.get_attributes_from_request(request_params))

        assert attributes["fi.span.kind"] == "EMBEDDING"
        assert attributes["llm.system"] == "together"

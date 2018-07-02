"""Tests for HuggingFace instrumentation."""
import json
from unittest.mock import MagicMock, patch

import pytest

from traceai_huggingface import HuggingFaceInstrumentor


class TestHuggingFaceInstrumentor:
    """Tests for HuggingFaceInstrumentor."""

    def test_instrumentation_dependencies(self):
        """Test instrumentation dependencies."""
        instrumentor = HuggingFaceInstrumentor()
        deps = instrumentor.instrumentation_dependencies()
        assert "huggingface-hub >= 0.20.0" in deps


class TestTextGenerationRequestAttributesExtractor:
    """Tests for text generation request attributes extraction."""

    def test_extract_text_generation_attributes(self):
        """Test extracting text generation request attributes."""
        from traceai_huggingface._request_attributes_extractor import _TextGenerationRequestAttributesExtractor

        extractor = _TextGenerationRequestAttributesExtractor()
        request_params = {
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "prompt": "The capital of France is",
            "max_new_tokens": 100,
            "temperature": 0.7,
        }

        attributes = dict(extractor.get_attributes_from_request(request_params))

        assert attributes["fi.span.kind"] == "LLM"
        assert attributes["llm.system"] == "huggingface"
        assert attributes["llm.model"] == "meta-llama/Llama-2-7b-chat-hf"

    def test_extract_generation_parameters(self):
        """Test extracting generation parameters."""
        from traceai_huggingface._request_attributes_extractor import _TextGenerationRequestAttributesExtractor

        extractor = _TextGenerationRequestAttributesExtractor()
        request_params = {
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "prompt": "Hello",
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50,
            "max_new_tokens": 200,
        }

        attributes = dict(extractor.get_attributes_from_request(request_params))

        assert "llm.invocation_parameters" in attributes
        params = json.loads(attributes["llm.invocation_parameters"])
        assert params["temperature"] == 0.8
        assert params["max_new_tokens"] == 200


class TestChatCompletionRequestAttributesExtractor:
    """Tests for chat completion request attributes extraction."""

    def test_extract_chat_attributes(self):
        """Test extracting chat completion request attributes."""
        from traceai_huggingface._request_attributes_extractor import _ChatCompletionRequestAttributesExtractor

        extractor = _ChatCompletionRequestAttributesExtractor()
        request_params = {
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
            "temperature": 0.7,
        }

        attributes = dict(extractor.get_attributes_from_request(request_params))

        assert attributes["fi.span.kind"] == "LLM"
        assert attributes["llm.system"] == "huggingface"

    def test_extract_messages(self):
        """Test extracting messages from request."""
        from traceai_huggingface._request_attributes_extractor import _ChatCompletionRequestAttributesExtractor

        extractor = _ChatCompletionRequestAttributesExtractor()
        request_params = {
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
            ],
        }

        extra_attrs = dict(extractor.get_extra_attributes_from_request(request_params))

        # Should have messages
        assert any("input_messages" in k for k in extra_attrs.keys())

    def test_extract_tools(self):
        """Test extracting tools from request."""
        from traceai_huggingface._request_attributes_extractor import _ChatCompletionRequestAttributesExtractor

        extractor = _ChatCompletionRequestAttributesExtractor()
        request_params = {
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather info",
                    },
                }
            ],
        }

        extra_attrs = dict(extractor.get_extra_attributes_from_request(request_params))

        assert extra_attrs.get("huggingface.tools_count") == 1


class TestFeatureExtractionRequestAttributesExtractor:
    """Tests for feature extraction request attributes extraction."""

    def test_extract_embedding_attributes(self):
        """Test extracting feature extraction request attributes."""
        from traceai_huggingface._request_attributes_extractor import _FeatureExtractionRequestAttributesExtractor

        extractor = _FeatureExtractionRequestAttributesExtractor()
        request_params = {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "inputs": "Hello, world!",
        }

        attributes = dict(extractor.get_attributes_from_request(request_params))


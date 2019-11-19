"""Tests for Ollama instrumentation."""
import json
from unittest.mock import MagicMock, patch

import pytest


class TestOllamaInstrumentor:
    """Tests for OllamaInstrumentor."""

    def test_instrumentation_dependencies(self):
        """Test instrumentation dependencies."""
        from traceai_ollama import OllamaInstrumentor

        instrumentor = OllamaInstrumentor()
        deps = instrumentor.instrumentation_dependencies()
        assert "ollama >= 0.3.0" in deps


class TestRequestAttributesExtractor:
    """Tests for request attributes extraction."""

    def test_extract_chat_attributes(self):
        """Test extracting chat request attributes."""
        from traceai_ollama._request_attributes_extractor import _RequestAttributesExtractor

        extractor = _RequestAttributesExtractor()
        request_params = {
            "model": "llama3.2",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
            "temperature": 0.7,
        }

        attributes = dict(extractor.get_attributes_from_request(request_params))

        assert attributes["fi.span.kind"] == "LLM"
        assert attributes["llm.system"] == "ollama"
        assert attributes["llm.model"] == "llama3.2"

    def test_extract_embed_attributes(self):
        """Test extracting embed request attributes."""
        from traceai_ollama._request_attributes_extractor import _RequestAttributesExtractor

        extractor = _RequestAttributesExtractor()
        request_params = {
            "model": "nomic-embed-text",
            "input": "Hello, world!",
            "_endpoint": "embed",
        }

        attributes = dict(extractor.get_attributes_from_request(request_params))

        assert attributes["fi.span.kind"] == "EMBEDDING"
        assert attributes["llm.system"] == "ollama"

    def test_extract_messages(self):
        """Test extracting message attributes."""
        from traceai_ollama._request_attributes_extractor import _RequestAttributesExtractor

        extractor = _RequestAttributesExtractor()
        request_params = {
            "model": "llama3.2",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
            ],
        }

        extra_attrs = dict(extractor.get_extra_attributes_from_request(request_params))

        assert "llm.input_messages.0.message.role" in extra_attrs or any(
            "message.role" in k for k in extra_attrs.keys()
        )


class TestResponseAttributesExtractor:
    """Tests for response attributes extraction."""

    def test_extract_chat_response(self):
        """Test extracting chat response attributes."""
        from traceai_ollama._response_attributes_extractor import _ResponseAttributesExtractor

        extractor = _ResponseAttributesExtractor()
        response = {
            "message": {"role": "assistant", "content": "Hello! How can I help?"},
            "prompt_eval_count": 10,
            "eval_count": 15,
            "total_duration": 1000000000,
        }

        attributes = dict(extractor.get_attributes(response))

        assert attributes.get("llm.token_count.prompt") == 10
        assert attributes.get("llm.token_count.completion") == 15

    def test_extract_generate_response(self):

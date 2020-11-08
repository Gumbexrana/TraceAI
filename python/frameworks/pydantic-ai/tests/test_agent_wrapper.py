"""Tests for Pydantic AI agent wrapper."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio


class TestSafeSerialize:
    """Test safe_serialize function."""

    def test_serialize_none(self):
        """Test serializing None."""
        from traceai_pydantic_ai._agent_wrapper import safe_serialize

        assert safe_serialize(None) == ""

    def test_serialize_string(self):
        """Test serializing string."""
        from traceai_pydantic_ai._agent_wrapper import safe_serialize

        assert safe_serialize("hello") == "hello"

    def test_serialize_dict(self):
        """Test serializing dict."""
        from traceai_pydantic_ai._agent_wrapper import safe_serialize

        result = safe_serialize({"key": "value"})
        assert "key" in result
        assert "value" in result

    def test_serialize_list(self):
        """Test serializing list."""
        from traceai_pydantic_ai._agent_wrapper import safe_serialize

        result = safe_serialize([1, 2, 3])
        assert "1" in result
        assert "2" in result
        assert "3" in result

    def test_serialize_truncates_long_string(self):
        """Test that long strings are truncated."""
        from traceai_pydantic_ai._agent_wrapper import safe_serialize

        long_string = "a" * 5000
        result = safe_serialize(long_string, max_length=100)
        assert len(result) == 100
        assert result.endswith("...")

    def test_serialize_complex_object(self):
        """Test serializing complex object."""
        from traceai_pydantic_ai._agent_wrapper import safe_serialize

        obj = {"nested": {"key": [1, 2, {"deep": "value"}]}}
        result = safe_serialize(obj)
        assert "nested" in result
        assert "deep" in result


class TestExtractModelInfo:
    """Test extract_model_info function."""

    def test_extract_string_model(self):
        """Test extracting model from string."""
        from traceai_pydantic_ai._agent_wrapper import extract_model_info

        agent = MagicMock()
        agent.model = "openai:gpt-4o"

        info = extract_model_info(agent)

        assert info["model_name"] == "openai:gpt-4o"
        assert info["model_provider"] == "openai"

    def test_extract_model_object(self):
        """Test extracting model from object with model_name."""
        from traceai_pydantic_ai._agent_wrapper import extract_model_info

        model = MagicMock()
        model.model_name = "claude-3-opus"

        agent = MagicMock()
        agent.model = model

        info = extract_model_info(agent)

        assert info["model_name"] == "claude-3-opus"
        assert info["model_provider"] == "anthropic"

    def test_extract_no_model(self):
        """Test extracting when no model."""
        from traceai_pydantic_ai._agent_wrapper import extract_model_info

        agent = MagicMock()
        agent.model = None

        info = extract_model_info(agent)

        assert info == {}


class TestExtractUsage:
    """Test extract_usage function."""

    def test_extract_usage_with_data(self):
        """Test extracting usage data."""
        from traceai_pydantic_ai._agent_wrapper import extract_usage

        usage = MagicMock()
        usage.request_tokens = 100
        usage.response_tokens = 50
        usage.total_tokens = 150
        usage.requests = 1

        result = MagicMock()
        result.usage = usage

        extracted = extract_usage(result)

        assert extracted["input_tokens"] == 100
        assert extracted["output_tokens"] == 50
        assert extracted["total_tokens"] == 150

    def test_extract_usage_none(self):
        """Test extracting when no usage."""
        from traceai_pydantic_ai._agent_wrapper import extract_usage

        result = MagicMock()
        result.usage = None

        extracted = extract_usage(result)
        assert extracted is None

    def test_extract_usage_alternate_field_names(self):
        """Test extracting with alternate field names."""
        from traceai_pydantic_ai._agent_wrapper import extract_usage

        usage = MagicMock()
        usage.request_tokens = None
        usage.input_tokens = 100
        usage.response_tokens = None
        usage.output_tokens = 50
        usage.total_tokens = 150


"""Extended tests for Pydantic AI agent wrapper."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
import json


class TestSafeSerializeEdgeCases:
    """Test edge cases for safe_serialize function."""

    def test_serialize_bytes(self):
        """Test serializing bytes."""
        from traceai_pydantic_ai._agent_wrapper import safe_serialize

        result = safe_serialize(b"hello bytes")
        assert "hello" in result or "bytes" in result

    def test_serialize_int(self):
        """Test serializing integer."""
        from traceai_pydantic_ai._agent_wrapper import safe_serialize

        assert safe_serialize(42) == "42"

    def test_serialize_float(self):
        """Test serializing float."""
        from traceai_pydantic_ai._agent_wrapper import safe_serialize

        assert safe_serialize(3.14) == "3.14"

    def test_serialize_bool(self):
        """Test serializing boolean."""
        from traceai_pydantic_ai._agent_wrapper import safe_serialize

        assert safe_serialize(True) == "True"
        assert safe_serialize(False) == "False"

    def test_serialize_nested_dict(self):
        """Test serializing nested dictionary."""
        from traceai_pydantic_ai._agent_wrapper import safe_serialize

        nested = {"level1": {"level2": {"level3": "deep"}}}
        result = safe_serialize(nested)
        assert "level1" in result
        assert "level2" in result
        assert "level3" in result
        assert "deep" in result

    def test_serialize_with_special_chars(self):
        """Test serializing with special characters."""
        from traceai_pydantic_ai._agent_wrapper import safe_serialize

        special = "Hello\nWorld\t\"quoted\""
        result = safe_serialize(special)
        assert result == special

    def test_serialize_unicode(self):
        """Test serializing unicode strings."""
        from traceai_pydantic_ai._agent_wrapper import safe_serialize

        unicode_str = "Hello 世界 🌍"
        result = safe_serialize(unicode_str)
        assert result == unicode_str

    def test_serialize_empty_dict(self):
        """Test serializing empty dict."""
        from traceai_pydantic_ai._agent_wrapper import safe_serialize

        assert safe_serialize({}) == "{}"

    def test_serialize_empty_list(self):
        """Test serializing empty list."""
        from traceai_pydantic_ai._agent_wrapper import safe_serialize

        assert safe_serialize([]) == "[]"

    def test_serialize_mixed_list(self):
        """Test serializing mixed type list."""
        from traceai_pydantic_ai._agent_wrapper import safe_serialize

        mixed = [1, "two", 3.0, {"four": 4}]
        result = safe_serialize(mixed)
        assert "1" in result
        assert "two" in result
        assert "3.0" in result
        assert "four" in result

    def test_serialize_class_instance(self):
        """Test serializing class instance."""
        from traceai_pydantic_ai._agent_wrapper import safe_serialize

        class MyClass:
            def __init__(self):
                self.value = 42

        obj = MyClass()
        result = safe_serialize(obj)
        assert "MyClass" in result or "42" in str(result)

    def test_serialize_truncation_boundary(self):
        """Test truncation at exact boundary."""
        from traceai_pydantic_ai._agent_wrapper import safe_serialize

        # Create string exactly at max length
        exactly_max = "a" * 100
        result = safe_serialize(exactly_max, max_length=100)
        assert len(result) == 100
        assert not result.endswith("...")

    def test_serialize_truncation_one_over(self):
        """Test truncation one over boundary."""
        from traceai_pydantic_ai._agent_wrapper import safe_serialize

        # Create string one over max length
        one_over = "a" * 101
        result = safe_serialize(one_over, max_length=100)
        assert len(result) == 100
        assert result.endswith("...")


class TestExtractModelInfoEdgeCases:
    """Test edge cases for extract_model_info function."""

    def test_model_with_class_name(self):
        """Test model with __class__.__name__."""
        from traceai_pydantic_ai._agent_wrapper import extract_model_info

        class FakeModel:
            pass

        agent = MagicMock()
        agent.model = FakeModel()

        info = extract_model_info(agent)
        # Should handle gracefully when no name attribute
        assert isinstance(info, dict)

    def test_model_with_empty_string(self):
        """Test model with empty string."""
        from traceai_pydantic_ai._agent_wrapper import extract_model_info

        agent = MagicMock()
        agent.model = ""

        info = extract_model_info(agent)
        # Empty string is falsy, so no model info is extracted
        assert info == {}

    def test_model_with_model_name_attr(self):
        """Test model object with model_name attribute."""
        from traceai_pydantic_ai._agent_wrapper import extract_model_info

        model = MagicMock()
        model.model_name = "gpt-4-turbo-preview"

        agent = MagicMock()
        agent.model = model

        info = extract_model_info(agent)
        assert info["model_name"] == "gpt-4-turbo-preview"
        assert info["model_provider"] == "openai"

    def test_model_with_name_attr(self):
        """Test model object with name attribute."""
        from traceai_pydantic_ai._agent_wrapper import extract_model_info

        model = MagicMock(spec=["name"])
        model.name = "claude-3-5-sonnet"
        # Mock that model_name doesn't exist
        del model.model_name

        agent = MagicMock()
        agent.model = model

        info = extract_model_info(agent)
        assert info["model_name"] == "claude-3-5-sonnet"


class TestExtractUsageEdgeCases:
    """Test edge cases for extract_usage function."""

    def test_usage_with_only_total_tokens(self):
        """Test usage with only total tokens."""
        from traceai_pydantic_ai._agent_wrapper import extract_usage

        usage = MagicMock()
        usage.request_tokens = None
        usage.response_tokens = None
        usage.input_tokens = None
        usage.output_tokens = None
        usage.total_tokens = 1000
        usage.requests = None

        result = MagicMock()
        result.usage = usage

        extracted = extract_usage(result)
        assert extracted["total_tokens"] == 1000

    def test_usage_with_zero_tokens(self):
        """Test usage with zero tokens."""
        from traceai_pydantic_ai._agent_wrapper import extract_usage

        # Create usage object with actual values (not MagicMock auto-values)
        class Usage:
            request_tokens = 0
            response_tokens = 0
            total_tokens = 0
            requests = 0

        result = MagicMock()
        result.usage = Usage()

        extracted = extract_usage(result)
        assert extracted["input_tokens"] == 0

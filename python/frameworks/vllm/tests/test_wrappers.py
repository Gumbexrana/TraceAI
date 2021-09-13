"""Tests for vLLM wrapper functions."""

import pytest
from unittest.mock import MagicMock, PropertyMock

from traceai_vllm._wrappers import (
    _WithTracer,
    _get_input_attributes,
    _get_output_attributes,
    _get_stream_output_attributes,
)
from fi_instrumentation.fi_types import FiSpanKindValues, SpanAttributes, MessageAttributes


class TestIsVLLMClient:
    """Tests for _is_vllm_client method in _WithTracer."""

    def test_returns_true_for_vllm_client(self, mock_vllm_client):
        """Should return True for client with vLLM base URL."""
        mock_tracer = MagicMock()
        wrapper = type("TestWrapper", (_WithTracer,), {})(
            tracer=mock_tracer,
            vllm_base_urls=["localhost:8000"],
        )
        assert wrapper._is_vllm_client(mock_vllm_client) is True

    def test_returns_false_for_non_vllm_client(self, mock_non_vllm_client):
        """Should return False for client with non-vLLM base URL."""
        mock_tracer = MagicMock()
        wrapper = type("TestWrapper", (_WithTracer,), {})(
            tracer=mock_tracer,
            vllm_base_urls=["localhost:8000"],
        )
        assert wrapper._is_vllm_client(mock_non_vllm_client) is False

    def test_returns_true_for_custom_vllm_url(self):
        """Should return True for custom vLLM URLs."""
        mock_tracer = MagicMock()
        wrapper = type("TestWrapper", (_WithTracer,), {})(
            tracer=mock_tracer,
            vllm_base_urls=["my-vllm-server.com:8080"],
        )

        mock_client = MagicMock()
        mock_client._client = MagicMock()
        type(mock_client._client).base_url = PropertyMock(
            return_value="http://my-vllm-server.com:8080/v1"
        )

        assert wrapper._is_vllm_client(mock_client) is True

    def test_returns_false_for_no_client(self):
        """Should return False when instance has no _client."""
        mock_tracer = MagicMock()
        wrapper = type("TestWrapper", (_WithTracer,), {})(
            tracer=mock_tracer,
            vllm_base_urls=["localhost:8000"],
        )
        instance = MagicMock(spec=[])
        assert wrapper._is_vllm_client(instance) is False


class TestGetInputAttributes:
    """Tests for _get_input_attributes function."""

    def test_basic_attributes(self):
        """Should include basic LLM attributes."""
        attrs = _get_input_attributes(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert attrs[SpanAttributes.GEN_AI_SPAN_KIND] == FiSpanKindValues.LLM.value
        assert attrs[SpanAttributes.GEN_AI_PROVIDER_NAME] == "vllm"
        assert attrs[SpanAttributes.GEN_AI_PROVIDER_NAME] == "vllm"
        assert attrs[SpanAttributes.GEN_AI_REQUEST_MODEL] == "meta-llama/Llama-2-7b-chat-hf"

    def test_message_attributes(self):
        """Should include message attributes."""
        attrs = _get_input_attributes(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        )

        assert attrs[f"{SpanAttributes.GEN_AI_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"] == "system"
        assert attrs[f"{SpanAttributes.GEN_AI_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"] == "You are helpful."
        assert attrs[f"{SpanAttributes.GEN_AI_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_ROLE}"] == "user"
        assert attrs[f"{SpanAttributes.GEN_AI_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_CONTENT}"] == "Hello"

    def test_optional_parameters(self):
        """Should include optional parameters when provided."""
        attrs = _get_input_attributes(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
        )

        assert SpanAttributes.GEN_AI_REQUEST_PARAMETERS in attrs

    def test_vllm_specific_parameters(self):
        """Should include vLLM-specific parameters."""
        attrs = _get_input_attributes(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[{"role": "user", "content": "Hello"}],
            best_of=3,
            use_beam_search=True,
        )

        assert SpanAttributes.GEN_AI_REQUEST_PARAMETERS in attrs
        import json
        params = json.loads(attrs[SpanAttributes.GEN_AI_REQUEST_PARAMETERS])
        assert params["best_of"] == 3
        assert params["use_beam_search"] is True

    def test_raw_input_included(self):
        """Should include raw input as JSON."""
        attrs = _get_input_attributes(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert SpanAttributes.INPUT_VALUE in attrs

    def test_handles_none_values(self):
        """Should skip None parameter values."""
        attrs = _get_input_attributes(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=None,
            temperature=None,
        )

        assert SpanAttributes.GEN_AI_REQUEST_PARAMETERS not in attrs


class TestGetOutputAttributes:
    """Tests for _get_output_attributes function."""

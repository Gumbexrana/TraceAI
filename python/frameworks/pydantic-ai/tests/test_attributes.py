"""Tests for Pydantic AI semantic attributes."""

import pytest


class TestPydanticAISpanKind:
    """Test PydanticAISpanKind enum."""

    def test_agent_run_value(self):
        """Test agent run span kind value."""
        from traceai_pydantic_ai._attributes import PydanticAISpanKind

        assert PydanticAISpanKind.AGENT_RUN.value == "agent_run"

    def test_model_request_value(self):
        """Test model request span kind value."""
        from traceai_pydantic_ai._attributes import PydanticAISpanKind

        assert PydanticAISpanKind.MODEL_REQUEST.value == "model_request"

    def test_tool_call_value(self):
        """Test tool call span kind value."""
        from traceai_pydantic_ai._attributes import PydanticAISpanKind

        assert PydanticAISpanKind.TOOL_CALL.value == "tool_call"

    def test_result_validation_value(self):
        """Test result validation span kind value."""
        from traceai_pydantic_ai._attributes import PydanticAISpanKind

        assert PydanticAISpanKind.RESULT_VALIDATION.value == "result_validation"

    def test_retry_value(self):
        """Test retry span kind value."""
        from traceai_pydantic_ai._attributes import PydanticAISpanKind

        assert PydanticAISpanKind.RETRY.value == "retry"

    def test_stream_value(self):
        """Test stream span kind value."""
        from traceai_pydantic_ai._attributes import PydanticAISpanKind

        assert PydanticAISpanKind.STREAM.value == "stream"


class TestPydanticAIAttributes:
    """Test PydanticAIAttributes class."""

    def test_span_kind_attribute(self):
        """Test span kind attribute."""
        from traceai_pydantic_ai._attributes import PydanticAIAttributes

        assert PydanticAIAttributes.SPAN_KIND == "pydantic_ai.span_kind"

    def test_agent_attributes(self):
        """Test agent-level attributes."""
        from traceai_pydantic_ai._attributes import PydanticAIAttributes

        assert PydanticAIAttributes.GEN_AI_AGENT_NAME == "pydantic_ai.agent.name"
        assert PydanticAIAttributes.AGENT_MODEL == "pydantic_ai.agent.model"
        assert PydanticAIAttributes.AGENT_INSTRUCTIONS == "pydantic_ai.agent.instructions"
        assert PydanticAIAttributes.AGENT_RESULT_TYPE == "pydantic_ai.agent.result_type"

    def test_run_attributes(self):
        """Test run-level attributes."""
        from traceai_pydantic_ai._attributes import PydanticAIAttributes

        assert PydanticAIAttributes.RUN_ID == "pydantic_ai.run.id"
        assert PydanticAIAttributes.RUN_METHOD == "pydantic_ai.run.method"
        assert PydanticAIAttributes.RUN_PROMPT == "pydantic_ai.run.prompt"
        assert PydanticAIAttributes.RUN_RESULT == "pydantic_ai.run.result"

    def test_tool_attributes(self):
        """Test tool-level attributes."""
        from traceai_pydantic_ai._attributes import PydanticAIAttributes

        assert PydanticAIAttributes.GEN_AI_TOOL_NAME == "pydantic_ai.tool.name"

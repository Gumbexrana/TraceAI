"""Extended tests for Pydantic AI semantic attributes."""

import pytest


class TestPydanticAISpanKindCompleteness:
    """Test PydanticAISpanKind enum completeness."""

    def test_all_span_kinds_have_unique_values(self):
        """Test all span kinds have unique values."""
        from traceai_pydantic_ai._attributes import PydanticAISpanKind

        values = [kind.value for kind in PydanticAISpanKind]
        assert len(values) == len(set(values))

    def test_span_kinds_are_strings(self):
        """Test span kinds are strings."""
        from traceai_pydantic_ai._attributes import PydanticAISpanKind

        for kind in PydanticAISpanKind:
            assert isinstance(kind.value, str)

    def test_span_kinds_are_lowercase(self):
        """Test span kinds are lowercase."""
        from traceai_pydantic_ai._attributes import PydanticAISpanKind

        for kind in PydanticAISpanKind:
            assert kind.value == kind.value.lower()


class TestPydanticAIAttributesCompleteness:
    """Test PydanticAIAttributes class completeness."""

    def test_all_attributes_are_strings(self):
        """Test all attribute values are strings."""
        from traceai_pydantic_ai._attributes import PydanticAIAttributes

        for attr_name in dir(PydanticAIAttributes):
            if not attr_name.startswith("_"):
                value = getattr(PydanticAIAttributes, attr_name)
                assert isinstance(value, str), f"{attr_name} is not a string"

    def test_pydantic_ai_attributes_have_prefix(self):
        """Test pydantic_ai specific attributes have correct prefix."""
        from traceai_pydantic_ai._attributes import PydanticAIAttributes

        pydantic_attrs = [
            PydanticAIAttributes.SPAN_KIND,
            PydanticAIAttributes.GEN_AI_AGENT_NAME,
            PydanticAIAttributes.AGENT_INSTRUCTIONS,
            PydanticAIAttributes.RUN_ID,
            PydanticAIAttributes.RUN_PROMPT,
            PydanticAIAttributes.GEN_AI_TOOL_NAME,
            PydanticAIAttributes.TOOL_ARGS,
        ]

        for attr in pydantic_attrs:
            assert attr.startswith("pydantic_ai.")

    def test_genai_attributes_follow_convention(self):
        """Test GenAI attributes follow semantic convention."""
        from traceai_pydantic_ai._attributes import PydanticAIAttributes

        genai_attrs = [
            PydanticAIAttributes.USAGE_INPUT_TOKENS,
            PydanticAIAttributes.USAGE_OUTPUT_TOKENS,
            PydanticAIAttributes.USAGE_TOTAL_TOKENS,
            PydanticAIAttributes.MODEL_NAME,
            PydanticAIAttributes.MODEL_PROVIDER,
        ]

        for attr in genai_attrs:
            assert attr.startswith("gen_ai.")

    def test_cost_attributes_exist(self):
        """Test cost tracking attributes exist."""
        from traceai_pydantic_ai._attributes import PydanticAIAttributes

        assert hasattr(PydanticAIAttributes, "COST_TOTAL_USD")
        assert hasattr(PydanticAIAttributes, "COST_INPUT_USD")
        assert hasattr(PydanticAIAttributes, "COST_OUTPUT_USD")

    def test_performance_attributes_exist(self):
        """Test performance tracking attributes exist."""
        from traceai_pydantic_ai._attributes import PydanticAIAttributes

        assert hasattr(PydanticAIAttributes, "DURATION_MS")
        assert hasattr(PydanticAIAttributes, "TIME_TO_FIRST_TOKEN_MS")

    def test_validation_attributes_exist(self):
        """Test validation attributes exist."""
        from traceai_pydantic_ai._attributes import PydanticAIAttributes

        assert hasattr(PydanticAIAttributes, "VALIDATION_IS_VALID")
        assert hasattr(PydanticAIAttributes, "VALIDATION_ERROR")
        assert hasattr(PydanticAIAttributes, "VALIDATION_RETRIES")

    def test_metadata_attributes_exist(self):
        """Test metadata attributes exist."""
        from traceai_pydantic_ai._attributes import PydanticAIAttributes

        assert hasattr(PydanticAIAttributes, "METADATA")
        assert hasattr(PydanticAIAttributes, "METADATA_TAGS")


class TestGetModelProviderExtended:
    """Extended tests for get_model_provider function."""

    def test_openai_variants(self):
        """Test various OpenAI model names."""
        from traceai_pydantic_ai._attributes import get_model_provider

        openai_models = [
            "gpt-4",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "o1-preview",
            "o1-mini",
            "o3-mini",
        ]
        for model in openai_models:
            assert get_model_provider(model) == "openai", f"Failed for {model}"

    def test_anthropic_variants(self):
        """Test various Anthropic model names."""
        from traceai_pydantic_ai._attributes import get_model_provider

        anthropic_models = [
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-haiku",
            "claude-3-5-sonnet",
            "claude-3-5-sonnet-20240620",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2",
        ]
        for model in anthropic_models:
            assert get_model_provider(model) == "anthropic", f"Failed for {model}"

    def test_google_variants(self):
        """Test various Google model names."""
        from traceai_pydantic_ai._attributes import get_model_provider


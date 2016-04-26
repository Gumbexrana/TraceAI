"""Tests for traceai_agno._attributes module."""

import pytest
from traceai_agno._attributes import (
    SpanAttributes,
    get_agent_attributes,
    get_tool_attributes,
    get_team_attributes,
    get_workflow_attributes,
    get_model_attributes,
    get_model_provider,
    create_trace_context,
    MODEL_PROVIDER_PATTERNS,
    _truncate,
)


class TestSpanAttributes:
    """Tests for SpanAttributes class."""

    def test_llm_attributes_exist(self):
        """Test LLM semantic attributes are defined."""
        assert SpanAttributes.GEN_AI_PROVIDER_NAME == "gen_ai.provider.name"
        assert SpanAttributes.GEN_AI_REQUEST_MODEL == "gen_ai.request.model"
        assert SpanAttributes.GEN_AI_RESPONSE_MODEL == "gen_ai.response.model"
        assert SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS == "gen_ai.request.max_tokens"
        assert SpanAttributes.GEN_AI_REQUEST_TEMPERATURE == "gen_ai.request.temperature"
        assert SpanAttributes.GEN_AI_REQUEST_TOP_P == "gen_ai.request.top_p"
        assert SpanAttributes.GEN_AI_USAGE_INPUT_TOKENS == "gen_ai.usage.input_tokens"
        assert SpanAttributes.GEN_AI_USAGE_OUTPUT_TOKENS == "gen_ai.usage.output_tokens"
        assert SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS == "gen_ai.usage.total_tokens"

    def test_agent_attributes_exist(self):
        """Test agent semantic attributes are defined."""
        assert SpanAttributes.GEN_AI_AGENT_NAME == "agent.name"
        assert SpanAttributes.AGENT_TYPE == "agent.type"
        assert SpanAttributes.AGENT_DESCRIPTION == "agent.description"
        assert SpanAttributes.AGENT_INSTRUCTIONS == "agent.instructions"

    def test_tool_attributes_exist(self):
        """Test tool semantic attributes are defined."""
        assert SpanAttributes.GEN_AI_TOOL_NAME == "gen_ai.tool.name"
        assert SpanAttributes.GEN_AI_TOOL_DESCRIPTION == "gen_ai.tool.description"
        assert SpanAttributes.TOOL_PARAMETERS == "gen_ai.tool.parameters"
        assert SpanAttributes.TOOL_RESULT == "gen_ai.tool.result"

    def test_agno_specific_attributes_exist(self):
        """Test Agno-specific attributes are defined."""
        assert SpanAttributes.AGNO_AGENT_ID == "agno.agent.id"
        assert SpanAttributes.AGNO_TOOL_COUNT == "agno.tool_count"
        assert SpanAttributes.AGNO_TEAM_NAME == "agno.team.name"
        assert SpanAttributes.AGNO_TEAM_MEMBERS == "agno.team.members"
        assert SpanAttributes.AGNO_WORKFLOW_NAME == "agno.workflow.name"
        assert SpanAttributes.AGNO_WORKFLOW_STEP == "agno.workflow.step"
        assert SpanAttributes.AGNO_SESSION_ID == "agno.session.id"
        assert SpanAttributes.AGNO_USER_ID == "agno.user.id"
        assert SpanAttributes.AGNO_DEBUG_MODE == "agno.debug_mode"
        assert SpanAttributes.AGNO_MARKDOWN == "agno.markdown"
        assert SpanAttributes.AGNO_MEMORY_ENABLED == "agno.memory.enabled"
        assert SpanAttributes.AGNO_KNOWLEDGE_ENABLED == "agno.knowledge.enabled"


class TestGetModelProvider:
    """Tests for get_model_provider function."""

    def test_openai_models(self):
        """Test OpenAI model detection."""
        assert get_model_provider("gpt-4") == "openai"
        assert get_model_provider("gpt-4-turbo") == "openai"
        assert get_model_provider("gpt-3.5-turbo") == "openai"
        assert get_model_provider("o1-preview") == "openai"
        assert get_model_provider("o3-mini") == "openai"
        assert get_model_provider("text-davinci-003") == "openai"
        assert get_model_provider("text-embedding-ada-002") == "openai"

    def test_anthropic_models(self):
        """Test Anthropic model detection."""
        assert get_model_provider("claude-3-5-sonnet-20241022") == "anthropic"
        assert get_model_provider("claude-3-opus") == "anthropic"
        assert get_model_provider("claude-2.1") == "anthropic"
        assert get_model_provider("claude-instant") == "anthropic"

    def test_google_models(self):
        """Test Google model detection."""
        assert get_model_provider("gemini-1.5-pro") == "google"
        assert get_model_provider("gemini-pro") == "google"
        assert get_model_provider("palm-2") == "google"
        assert get_model_provider("text-bison") == "google"

    def test_mistral_models(self):
        """Test Mistral model detection."""
        assert get_model_provider("mistral-large") == "mistral"
        assert get_model_provider("mistral-medium") == "mistral"
        assert get_model_provider("mixtral-8x7b") == "mistral"
        assert get_model_provider("codestral") == "mistral"

    def test_meta_models(self):
        """Test Meta model detection."""
        assert get_model_provider("llama-3.1-70b") == "meta"
        assert get_model_provider("llama-2-70b") == "meta"

    def test_cohere_models(self):
        """Test Cohere model detection."""
        assert get_model_provider("command-r-plus") == "cohere"
        assert get_model_provider("command-r") == "cohere"
        assert get_model_provider("embed-english-v3.0") == "cohere"

    def test_deepseek_models(self):
        """Test DeepSeek model detection."""
        assert get_model_provider("deepseek-chat") == "deepseek"
        assert get_model_provider("deepseek-coder") == "deepseek"

    def test_ollama_models(self):
        """Test Ollama model detection."""
        assert get_model_provider("ollama/llama3") == "ollama"
        assert get_model_provider("ollama/mistral") == "ollama"
        assert get_model_provider("ollama/codellama") == "ollama"

    def test_groq_models(self):
        """Test Groq model detection."""
        assert get_model_provider("groq/llama3-70b") == "groq"
        assert get_model_provider("groq/mixtral-8x7b") == "groq"

    def test_together_models(self):
        """Test Together AI model detection."""
        assert get_model_provider("together/llama-2-70b") == "together"
        assert get_model_provider("together/mixtral-8x7b") == "together"

    def test_fireworks_models(self):
        """Test Fireworks model detection."""
        assert get_model_provider("fireworks/llama-v3-70b") == "fireworks"

    def test_bedrock_models(self):
        """Test AWS Bedrock model detection."""
        assert get_model_provider("amazon.titan-text-express") == "bedrock"
        assert get_model_provider("ai21.jamba-instruct") == "bedrock"
        assert get_model_provider("stability.stable-diffusion-xl") == "bedrock"

    def test_azure_models(self):
        """Test Azure OpenAI model detection."""
        assert get_model_provider("azure/gpt-4") == "azure"
        assert get_model_provider("azure/gpt-35-turbo") == "azure"

    def test_explicit_prefix_detection(self):
        """Test that explicit prefixes are properly detected."""
        assert get_model_provider("openai/gpt-4") == "openai"
        assert get_model_provider("anthropic/claude-3") == "anthropic"
        assert get_model_provider("google/gemini-pro") == "google"
        assert get_model_provider("mistral/mistral-large") == "mistral"

    def test_unknown_models(self):
        """Test unknown model detection."""
        assert get_model_provider("unknown-model") == "unknown"
        assert get_model_provider("custom-llm") == "unknown"
        assert get_model_provider("") == "unknown"
        assert get_model_provider(None) == "unknown"


class TestGetAgentAttributes:
    """Tests for get_agent_attributes function."""

    def test_basic_agent_attributes(self, mock_agent):
        """Test basic agent attribute extraction."""
        attrs = get_agent_attributes(mock_agent)

        assert attrs[SpanAttributes.AGENT_TYPE] == "MockAgent"
        assert attrs[SpanAttributes.GEN_AI_AGENT_NAME] == "TestAgent"
        assert attrs[SpanAttributes.AGNO_AGENT_ID] == "agent-123"
        assert attrs[SpanAttributes.AGENT_DESCRIPTION] == "A test agent"
        assert attrs[SpanAttributes.AGENT_INSTRUCTIONS] == "Be helpful and accurate"

    def test_model_attributes(self, mock_agent):
        """Test model attribute extraction from agent."""
        attrs = get_agent_attributes(mock_agent)

        assert attrs[SpanAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4"
        assert attrs[SpanAttributes.GEN_AI_PROVIDER_NAME] == "openai"

    def test_tool_count(self, mock_agent_with_tools):
        """Test tool count extraction."""
        attrs = get_agent_attributes(mock_agent_with_tools)

        assert attrs[SpanAttributes.AGNO_TOOL_COUNT] == 2

    def test_debug_mode(self, mock_agent):
        """Test debug mode extraction."""
        attrs = get_agent_attributes(mock_agent)

        assert attrs[SpanAttributes.AGNO_DEBUG_MODE] is False

    def test_markdown_setting(self, mock_agent):
        """Test markdown setting extraction."""
        attrs = get_agent_attributes(mock_agent)

        assert attrs[SpanAttributes.AGNO_MARKDOWN] is True

    def test_memory_enabled(self, mock_agent):
        """Test memory enabled detection."""
        attrs = get_agent_attributes(mock_agent)

        assert attrs[SpanAttributes.AGNO_MEMORY_ENABLED] is False


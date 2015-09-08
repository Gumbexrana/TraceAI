"""
E2E Tests for LangChain Instrumentation

Tests LangChain chains, agents, and tools with tracing.
"""

import pytest
import os
import time
from typing import Dict, Any

from config import config, skip_if_no_openai


@pytest.fixture(scope="module")
def setup_langchain():
    """Set up LangChain with instrumentation."""
    if not config.has_openai():
        pytest.skip("OpenAI API key required for LangChain tests")

    os.environ["OPENAI_API_KEY"] = config.openai_api_key

    # Import and instrument
    from fi_instrumentation import register
    try:
        from traceai_langchain import LangChainInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_langchain not installed or incompatible")

    # Register tracer
    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
    )

    # Instrument LangChain
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    yield

    # Cleanup
    LangChainInstrumentor().uninstrument()


@skip_if_no_openai
class TestLangChainLLM:
    """Test LangChain LLM integration."""

    def test_basic_llm_invoke(self, setup_langchain):
        """Test basic LLM invocation."""
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=50)
        response = llm.invoke("Say hello in one word.")

        assert response.content is not None
        time.sleep(2)
        print(f"Response: {response.content}")

    def test_llm_with_system_message(self, setup_langchain):
        """Test LLM with system message."""
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=30)
        messages = [
            SystemMessage(content="You respond in exactly one word."),
            HumanMessage(content="What is 2+2?"),
        ]
        response = llm.invoke(messages)

        assert response.content is not None

    def test_llm_streaming(self, setup_langchain):
        """Test LLM streaming."""
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=50, streaming=True)

        chunks = []
        for chunk in llm.stream("Count from 1 to 5."):
            if chunk.content:
                chunks.append(chunk.content)

        assert len(chunks) > 0
        print(f"Streamed: {''.join(chunks)}")


@skip_if_no_openai
class TestLangChainChains:
    """Test LangChain chains."""

    def test_simple_chain(self, setup_langchain):
        """Test simple LCEL chain."""
        from langchain_openai import ChatOpenAI

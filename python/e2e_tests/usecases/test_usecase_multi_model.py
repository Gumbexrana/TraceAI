"""
E2E Test: Multi-Model Pipeline Use Case

Simulates a pipeline using multiple LLM providers.
"""

import pytest
import os
import time
from typing import Dict, Any

from config import config, skip_if_no_openai, skip_if_no_anthropic


@pytest.fixture(scope="module")
def setup_multi_model():
    """Set up multiple providers with instrumentation."""
    # Set API keys
    if config.has_openai():
        os.environ["OPENAI_API_KEY"] = config.openai_api_key
    if config.has_anthropic():
        os.environ["ANTHROPIC_API_KEY"] = config.anthropic_api_key

    from fi_instrumentation import register
    from traceai_openai import OpenAIInstrumentor
    from traceai_anthropic import AnthropicInstrumentor

    tracer_provider = register(
        project_name="e2e_multi_model_usecase",
        project_version_name="1.0.0",
        verbose=False,
    )

    if config.has_openai():
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    if config.has_anthropic():
        AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)

    yield tracer_provider

    if config.has_openai():
        OpenAIInstrumentor().uninstrument()
    if config.has_anthropic():
        AnthropicInstrumentor().uninstrument()


@pytest.mark.skipif(
    not (config.has_openai() and config.has_anthropic()),
    reason="Both OpenAI and Anthropic keys required"
)
class TestMultiModelPipeline:
    """Test multi-model pipeline scenarios."""

    def test_draft_and_refine(self, setup_multi_model):
        """Use one model to draft, another to refine."""
        from openai import OpenAI
        from anthropic import Anthropic

        openai_client = OpenAI()
        anthropic_client = Anthropic()

        # Step 1: OpenAI generates initial draft
        draft_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Write a short paragraph about the benefits of exercise."
                }
            ],
            max_tokens=150,
        )
        draft = draft_response.choices[0].message.content
        print(f"Draft (OpenAI): {draft}")

        # Step 2: Anthropic refines the draft
        refine_response = anthropic_client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": f"Improve this paragraph to be more engaging and concise:\n\n{draft}"
                }
            ],
        )
        refined = refine_response.content[0].text
        print(f"Refined (Anthropic): {refined}")

        assert len(draft) > 0
        assert len(refined) > 0

        time.sleep(2)


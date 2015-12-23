"""
E2E Test: Customer Support Chatbot Use Case

Simulates a multi-turn conversation chatbot with session tracking.
"""

import pytest
import os
import time
import uuid
from typing import List, Dict, Any

from config import config, skip_if_no_openai


@pytest.fixture(scope="module")
def setup_chatbot():
    """Set up OpenAI with instrumentation for chatbot."""
    if not config.has_openai():
        pytest.skip("OpenAI API key required")

    os.environ["OPENAI_API_KEY"] = config.openai_api_key

    from fi_instrumentation import register
    from traceai_openai import OpenAIInstrumentor

    tracer_provider = register(
        project_name="e2e_chatbot_usecase",
        project_version_name="1.0.0",
        verbose=False,
    )

    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

    from openai import OpenAI
    client = OpenAI()

    yield client

    OpenAIInstrumentor().uninstrument()


class ChatbotSession:
    """Simulates a chatbot session with conversation history."""

    def __init__(self, client, session_id: str, user_id: str):
        self.client = client
        self.session_id = session_id
        self.user_id = user_id
        self.messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": """You are a helpful customer support assistant for TechCorp.
                Be concise and helpful. If you don't know something, say so."""
            }
        ]

    def send_message(self, user_message: str) -> str:
        """Send a message and get response."""
        self.messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages,
            max_tokens=150,
        )

        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})


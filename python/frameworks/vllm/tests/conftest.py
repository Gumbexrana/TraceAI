"""Pytest fixtures for vLLM tests."""

from unittest.mock import MagicMock, PropertyMock
import pytest


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI-compatible response."""
    response = MagicMock()
    response.id = "chatcmpl-123"
    response.model = "meta-llama/Llama-2-7b-chat-hf"
    response.choices = [MagicMock()]
    response.choices[0].message = MagicMock()
    response.choices[0].message.role = "assistant"
    response.choices[0].message.content = "Hello! How can I help you?"
    response.choices[0].finish_reason = "stop"
    response.usage = MagicMock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 8
    response.usage.total_tokens = 18
    response.model_dump = MagicMock(return_value={
        "id": "chatcmpl-123",
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "choices": [{"message": {"role": "assistant", "content": "Hello! How can I help you?"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    })
    return response


@pytest.fixture
def mock_stream_chunks():
    """Create mock streaming chunks."""
    chunks = []
    for i, content in enumerate(["Hello", "!", " How", " can", " I", " help", "?"]):
        chunk = MagicMock()
        chunk.id = "chatcmpl-123"
        chunk.model = "meta-llama/Llama-2-7b-chat-hf"
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = content
        chunk.choices[0].finish_reason = None if i < 6 else "stop"
        chunk.usage = None
        chunk.model_dump = MagicMock(return_value={"id": f"chunk-{i}", "content": content})
        chunks.append(chunk)

    # Last chunk has usage
    chunks[-1].usage = MagicMock()
    chunks[-1].usage.prompt_tokens = 10

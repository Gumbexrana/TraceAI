"""
E2E Tests for HuggingFace SDK Instrumentation

Tests HuggingFace InferenceClient with tracing. Requires HF_API_KEY.
"""

import pytest
import time

from config import config, skip_if_no_hf


@pytest.fixture(scope="module")
def hf_client():
    """Create an instrumented HuggingFace client."""
    if not config.has_hf():
        pytest.skip("HF_API_KEY not set")

    from fi_instrumentation import register
    try:
        from traceai_huggingface import HuggingFaceInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_huggingface not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    HuggingFaceInstrumentor().instrument(tracer_provider=tracer_provider)

    from huggingface_hub import InferenceClient

    client = InferenceClient(token=config.hf_api_key)

    yield client

    HuggingFaceInstrumentor().uninstrument()


@skip_if_no_hf
class TestHuggingFaceChatCompletion:
    """Test HuggingFace chat completion instrumentation."""

    def test_basic_chat(self, hf_client):
        """Test basic chat completion."""

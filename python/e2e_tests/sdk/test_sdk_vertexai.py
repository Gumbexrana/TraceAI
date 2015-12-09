"""
E2E Tests for Vertex AI SDK Instrumentation

Tests Vertex AI instrumentation. Requires GCP project and credentials.
"""

import pytest
import time

from config import config, skip_if_no_vertexai


@pytest.fixture(scope="module")
def vertexai_model():
    """Create an instrumented Vertex AI model."""
    if not config.has_vertexai():
        pytest.skip("Google Cloud not configured")

    from fi_instrumentation import register
    try:
        from traceai_vertexai import VertexAIInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_vertexai not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    VertexAIInstrumentor().instrument(tracer_provider=tracer_provider)

    import vertexai
    from vertexai.generative_models import GenerativeModel

    vertexai.init(project=config.google_cloud_project)
    model = GenerativeModel("gemini-2.0-flash")

    yield model

    VertexAIInstrumentor().uninstrument()


@skip_if_no_vertexai
class TestVertexAIGenerateContent:
    """Test Vertex AI generate_content instrumentation."""

    def test_basic_generate(self, vertexai_model):
        """Test basic content generation."""
        response = vertexai_model.generate_content(
            "Say 'Hello E2E Test' in exactly 3 words."
        )

        assert response.text is not None
        time.sleep(2)
        print(f"Response: {response.text}")

    def test_streaming(self, vertexai_model):
        """Test streaming content generation."""
        chunks = []

        response = vertexai_model.generate_content(
            "Count from 1 to 5.",

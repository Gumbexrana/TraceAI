"""
E2E Tests for Instructor SDK Instrumentation

Tests Instructor instrumentation using Google's OpenAI-compatible endpoint.
Instructor wraps OpenAI client for structured extraction.
"""

import pytest
import time

from config import config, skip_if_no_google


@pytest.fixture(scope="module")
def instructor_client():
    """Create an instrumented Instructor client."""
    if not config.has_google():
        pytest.skip("GOOGLE_API_KEY not set")

    from fi_instrumentation import register
    try:
        from traceai_instructor import InstructorInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_instructor not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    InstructorInstrumentor().instrument(tracer_provider=tracer_provider)

    from openai import OpenAI
    import instructor

    openai_client = OpenAI(
        base_url=config.google_openai_base_url,
        api_key=config.google_api_key,
    )

    client = instructor.from_openai(openai_client)

    yield client

    InstructorInstrumentor().uninstrument()


@skip_if_no_google
class TestInstructorExtraction:
    """Test Instructor structured extraction."""

    def test_simple_extraction(self, instructor_client):
        """Test simple structured extraction."""
        from pydantic import BaseModel

        class UserInfo(BaseModel):
            name: str
            age: int

        response = instructor_client.chat.completions.create(
            model=config.google_model,
            messages=[
                {"role": "user", "content": "My name is John and I am 30 years old."}
            ],
            response_model=UserInfo,
        )

        assert isinstance(response, UserInfo)
        assert response.name == "John"
        assert response.age == 30

        time.sleep(2)

"""
E2E Tests for PII Redaction via TraceConfig

Tests that pii_redaction=True on TraceConfig replaces PII patterns
in traced span attributes with <ENTITY_TYPE> tokens before export.

Uses Google GenAI (requires GOOGLE_API_KEY).
"""

import time

import pytest

from fi_instrumentation.fi_types import ProjectType

from config import config, skip_if_no_google


PII_PROMPT = (
    "Hi, my name is John Smith. My email is john.smith@acme-corp.com "
    "and my phone number is (555) 867-5309. "
    "My SSN is 123-45-6789 and my credit card is 4111 1111 1111 1111. "
    "I'm connecting from IP 192.168.1.42. "
    "My API key is sk-live-ABCDEFGHIJKLMNOPQRSTuvwx. "
    "Can you help me with my account?"
)


@pytest.fixture(scope="module")
def pii_genai_client():
    """Create an instrumented Google GenAI client with PII redaction ON."""
    if not config.has_google():
        pytest.skip("GOOGLE_API_KEY not set")

    from fi_instrumentation import register, TraceConfig

    try:
        from traceai_google_genai import GoogleGenAIInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_google_genai not installed or incompatible")

    tracer_provider = register(
        project_name="pii-redaction-e2e-test",
        project_type=ProjectType.OBSERVE,
        verbose=False,
    )

    GoogleGenAIInstrumentor().instrument(
        tracer_provider=tracer_provider,
        config=TraceConfig(pii_redaction=True),
    )

    from google import genai

    client = genai.Client(api_key=config.google_api_key)

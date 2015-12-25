"""
E2E Test: Structured Data Extraction Use Case

Simulates extracting structured data from unstructured text.
"""

import pytest
import os
import time
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass

from config import config, skip_if_no_openai


@pytest.fixture(scope="module")
def setup_extraction():
    """Set up OpenAI with instrumentation for extraction."""
    if not config.has_openai():
        pytest.skip("OpenAI API key required")

    os.environ["OPENAI_API_KEY"] = config.openai_api_key

    from fi_instrumentation import register
    from traceai_openai import OpenAIInstrumentor

    tracer_provider = register(
        project_name="e2e_extraction_usecase",
        project_version_name="1.0.0",
        verbose=False,
    )

    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

    from openai import OpenAI
    client = OpenAI()

    yield client

    OpenAIInstrumentor().uninstrument()


@skip_if_no_openai
class TestEntityExtraction:
    """Test entity extraction use cases."""

    def test_extract_person_info(self, setup_extraction):
        """Extract person information from text."""
        text = """
        John Smith is a 35-year-old software engineer from San Francisco.
        He can be reached at john.smith@email.com or (555) 123-4567.
        He has been working at TechCorp for 5 years.
        """

        response = setup_extraction.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Extract structured information from text. Return valid JSON."
                },
                {
                    "role": "user",
                    "content": f"""Extract the following information from this text:
                    - Name
                    - Age
                    - Occupation
                    - Location
                    - Email
                    - Phone
                    - Company
                    - Years at company

                    Text: {text}

                    Return as JSON."""
                }
            ],
            max_tokens=300,
            response_format={"type": "json_object"},
        )

        result = response.choices[0].message.content
        print(f"Extracted: {result}")

        # Verify it's valid JSON
        data = json.loads(result)
        assert "name" in data.get("Name", data).lower() or "john" in str(data).lower()

        time.sleep(2)

    def test_extract_product_info(self, setup_extraction):
        """Extract product information from description."""
        text = """
        The new iPhone 15 Pro Max features a 6.7-inch Super Retina XDR display,
        A17 Pro chip, and starts at $1,199. It comes in Natural Titanium,
        Blue Titanium, White Titanium, and Black Titanium colors.
        Storage options include 256GB, 512GB, and 1TB.
        """

        response = setup_extraction.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Extract product details. Return valid JSON."
                },
                {
                    "role": "user",
                    "content": f"""Extract product details:
                    - Product name
                    - Screen size
                    - Processor
                    - Starting price
                    - Available colors

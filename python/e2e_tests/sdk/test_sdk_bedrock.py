"""
E2E Tests for AWS Bedrock SDK Instrumentation

Tests Bedrock instrumentation via boto3. Requires AWS credentials.
"""

import pytest
import json
import time

from config import config, skip_if_no_bedrock


@pytest.fixture(scope="module")
def bedrock_client():
    """Create an instrumented Bedrock client."""
    if not config.has_bedrock():
        pytest.skip("AWS credentials not set")

    from fi_instrumentation import register
    try:
        from traceai_bedrock import BedrockInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_bedrock not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

    BedrockInstrumentor().instrument(tracer_provider=tracer_provider)

    import boto3

    client = boto3.client(
        "bedrock-runtime",
        region_name=config.aws_bedrock_region,
        aws_access_key_id=config.aws_access_key_id,
        aws_secret_access_key=config.aws_secret_access_key,
    )

    yield client

    BedrockInstrumentor().uninstrument()


@skip_if_no_bedrock
class TestBedrockConverse:
    """Test Bedrock Converse API instrumentation."""

    def test_basic_converse(self, bedrock_client):
        """Test basic converse call."""
        response = bedrock_client.converse(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"text": "Say 'Hello E2E Test' in exactly 3 words."}
                    ],
                }
            ],
            inferenceConfig={"maxTokens": 50},
        )

        assert response["output"]["message"]["content"][0]["text"] is not None
        time.sleep(2)

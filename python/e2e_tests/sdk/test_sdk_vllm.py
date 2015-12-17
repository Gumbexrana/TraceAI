"""
E2E Tests for vLLM SDK Instrumentation

Tests vLLM instrumentation. Requires a running vLLM server.
Set VLLM_BASE_URL to enable.
"""

import pytest
import os
import time

from config import config


VLLM_BASE_URL = os.getenv("VLLM_BASE_URL")
VLLM_MODEL = os.getenv("VLLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
HAS_VLLM = bool(VLLM_BASE_URL)

skip_if_no_vllm = pytest.mark.skipif(
    not HAS_VLLM, reason="VLLM_BASE_URL not set"
)


@pytest.fixture(scope="module")
def vllm_client():
    """Create an instrumented vLLM client via OpenAI-compat."""
    if not HAS_VLLM:
        pytest.skip("VLLM_BASE_URL not set")

    from fi_instrumentation import register
    try:
        from traceai_vllm import VLLMInstrumentor
    except (ImportError, AttributeError):
        pytest.skip("traceai_vllm not installed or incompatible")

    tracer_provider = register(
        project_name=config.project_name,
        project_version_name=config.project_version_name,
        project_type=config.project_type,
        verbose=False,
    )

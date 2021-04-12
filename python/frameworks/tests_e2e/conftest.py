"""
Pytest configuration for E2E integration tests.

These tests require real database instances. Use the provided docker-compose.yml
to spin up test databases.
"""

import pytest
from typing import Generator

from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires real DB)"
    )
    config.addinivalue_line(
        "markers", "chromadb: mark test as requiring ChromaDB"
    )
    config.addinivalue_line(
        "markers", "qdrant: mark test as requiring Qdrant"
    )

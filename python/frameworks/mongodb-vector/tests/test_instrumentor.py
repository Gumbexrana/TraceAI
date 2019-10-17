"""Unit tests for MongoDB instrumentation."""

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture
def in_memory_span_exporter():
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(in_memory_span_exporter):
    resource = Resource(attributes={"service.name": "test-mongodb"})

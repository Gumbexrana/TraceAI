"""Tests for the span-processor / exporter / provider tuning exposed by the SDK.

These verify that the tuning params (previously silently dropped) are now accepted
and forwarded, and that env vars / upstream defaults still apply when unset.

Helpers below read upstream OTel internals in a version-tolerant way: OTel SDK
>=1.34 delegates to a `_batch_processor`, older versions store the values directly
on the processor.
"""

import os
import warnings
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import SpanLimits
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

from fi_instrumentation.fi_types import ProjectType
from fi_instrumentation.otel import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    TracerProvider,
    Transport,
    register,
)


def _bp_get(processor, name):
    """Read a batch-processor setting across OTel layouts."""
    holder = getattr(processor, "_batch_processor", processor)
    for obj, attr in ((holder, "_" + name), (processor, name), (processor, "_" + name)):
        if hasattr(obj, attr):
            return getattr(obj, attr)
    raise AttributeError(name)


def _queue_batch(processor):
    return _bp_get(processor, "max_queue_size"), _bp_get(processor, "max_export_batch_size")


def _exporter_of(processor):
    """Effective exporter across OTel layouts (batch and simple processors)."""
    bp = getattr(processor, "_batch_processor", None)
    if bp is not None and hasattr(bp, "_exporter"):
        return bp._exporter
    return getattr(processor, "span_exporter", None)


def _clear_env(keys):
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)


@pytest.fixture
def clean_bsp_env():
    yield from _clear_env(
        [
            "OTEL_BSP_MAX_QUEUE_SIZE",
            "OTEL_BSP_MAX_EXPORT_BATCH_SIZE",
            "OTEL_BSP_SCHEDULE_DELAY",
            "OTEL_BSP_EXPORT_TIMEOUT",
        ]
    )


@pytest.fixture
def clean_exporter_env():
    yield from _clear_env(
        ["OTEL_EXPORTER_OTLP_TIMEOUT", "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT"]
    )


class TestBatchSpanProcessorTuning:
    def test_defaults_match_upstream(self, clean_bsp_env):
        """No kwargs, no env -> upstream OTel defaults (unchanged)."""
        p = BatchSpanProcessor(span_exporter=InMemorySpanExporter())
        assert _queue_batch(p) == (2048, 512)

    def test_explicit_kwargs_are_forwarded(self, clean_bsp_env):
        """The core fix: kwargs are now honored instead of dropped."""
        p = BatchSpanProcessor(
            span_exporter=InMemorySpanExporter(),
            max_queue_size=10000,
            max_export_batch_size=1024,
        )
        assert _queue_batch(p) == (10000, 1024)

    def test_schedule_delay_and_export_timeout_forwarded(self, clean_bsp_env):
        p = BatchSpanProcessor(
            span_exporter=InMemorySpanExporter(),
            schedule_delay_millis=1234,
            export_timeout_millis=5678,
        )
        assert _bp_get(p, "schedule_delay_millis") == 1234
        assert _bp_get(p, "export_timeout_millis") == 5678

    def test_env_vars_respected_when_unset(self, clean_bsp_env):
        os.environ["OTEL_BSP_MAX_QUEUE_SIZE"] = "3333"
        p = BatchSpanProcessor(span_exporter=InMemorySpanExporter())
        queue, batch = _queue_batch(p)
        assert queue == 3333
        assert batch == 512  # untouched -> upstream default

    def test_env_var_batch_size_respected_when_unset(self, clean_bsp_env):
        os.environ["OTEL_BSP_MAX_EXPORT_BATCH_SIZE"] = "128"
        p = BatchSpanProcessor(span_exporter=InMemorySpanExporter())
        assert _queue_batch(p)[1] == 128

    def test_explicit_kwarg_overrides_env(self, clean_bsp_env):
        os.environ["OTEL_BSP_MAX_QUEUE_SIZE"] = "3333"
        p = BatchSpanProcessor(
            span_exporter=InMemorySpanExporter(), max_queue_size=9000
        )
        assert _queue_batch(p)[0] == 9000


class TestExporterTuning:
    """timeout must reach the default OTLP exporter (both processors)."""

    def test_batch_forwards_timeout(self):
        p = BatchSpanProcessor(timeout=42)
        assert _exporter_of(p)._timeout == 42

    def test_simple_forwards_timeout(self):
        p = SimpleSpanProcessor(timeout=7)
        assert p.span_exporter._timeout == 7

    def test_unset_timeout_uses_upstream_default(self, clean_exporter_env):
        """Backward-compat: no timeout -> upstream default (10s)."""
        p = BatchSpanProcessor(span_exporter=None)
        assert _exporter_of(p)._timeout == 10

    def test_prebuilt_span_exporter_is_used_as_is(self):
        """A caller-supplied exporter is used verbatim; timeout ignored."""
        exp = InMemorySpanExporter()
        p = BatchSpanProcessor(span_exporter=exp, timeout=99)
        assert _exporter_of(p) is exp


class TestRegisterPlumbsTuning:
    """register() must forward the tuning kwargs to the processor / provider."""

    @patch("fi_instrumentation.otel.check_custom_eval_config_exists", return_value=False)

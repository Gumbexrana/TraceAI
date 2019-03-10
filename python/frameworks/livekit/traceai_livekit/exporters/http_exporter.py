"""
HTTP exporter for LiveKit integration with Future AGI.

This module provides an HTTP span exporter that maps LiveKit attributes
to Future AGI conventions.
"""

from typing import Any

from .base_exporter import BaseMappedSpanExporter

try:
    # Use FI's HTTP exporter as base
    from fi_instrumentation.otel import HTTPSpanExporter as _FIHTTPSpanExporter
except ImportError:
    _FIHTTPSpanExporter = None


class MappedHTTPSpanExporter(BaseMappedSpanExporter, _FIHTTPSpanExporter):
    """HTTP exporter that maps LiveKit attributes to Future AGI conventions."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the mapped HTTP exporter."""

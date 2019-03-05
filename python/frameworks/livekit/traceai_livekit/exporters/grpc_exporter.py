"""
gRPC exporter for LiveKit integration with Future AGI.

This module provides a gRPC span exporter that maps LiveKit attributes
to Future AGI conventions.
"""

from typing import Any

from .base_exporter import BaseMappedSpanExporter

try:
    # Use FI's gRPC exporter as base
    from fi_instrumentation.otel import GRPCSpanExporter as _FIGRPCSpanExporter
except ImportError:
    _FIGRPCSpanExporter = None


class MappedGRPCSpanExporter(BaseMappedSpanExporter, _FIGRPCSpanExporter):
    """gRPC exporter that maps LiveKit attributes to Future AGI conventions."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the mapped gRPC exporter."""

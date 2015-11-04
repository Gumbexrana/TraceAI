"""
E2E Tests for Pipecat Mapper/Exporter

Tests Pipecat span mapper and exporter creation.
Pipecat is an attribute mapper, not a full instrumentor — tests verify
exporter creation and attribute mapping logic.
"""

import pytest
import time

from config import config


class TestPipecatExporter:
    """Test Pipecat exporter creation and attribute mapping."""

    def test_import_and_create_exporter(self):
        """Test that Pipecat exporter can be imported and created."""
        try:
            from traceai_pipecat import PipecatSpanExporter

            assert PipecatSpanExporter is not None

"""
E2E Tests for LiveKit Mapper/Exporter

Tests LiveKit span mapper and exporter creation.
LiveKit is an attribute mapper, not a full instrumentor — tests verify
exporter creation and attribute mapping logic.
"""

import pytest
import time

from config import config


class TestLiveKitExporter:
    """Test LiveKit exporter creation and attribute mapping."""

    def test_import_and_create_exporter(self):
        """Test that LiveKit exporter can be imported and created."""
        try:
            from traceai_livekit import LiveKitSpanExporter

            assert LiveKitSpanExporter is not None

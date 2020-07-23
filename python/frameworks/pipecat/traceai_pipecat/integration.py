"""
Integration functions for Pipecat with Future AGI.

This module provides functions to install attribute mapping for Pipecat applications
by updating existing span exporters in the tracer provider.
"""

import logging
from typing import Optional

from fi_instrumentation.otel import Transport
from opentelemetry import trace as trace_api

from .exporters import MappedGRPCSpanExporter, MappedHTTPSpanExporter

logger = logging.getLogger(__name__)


def enable_fi_attribute_mapping(transport: Transport = Transport.HTTP) -> bool:
    """
    Install the mapping exporter by swapping FI's exporter instances.

    This function finds existing span exporters in the global tracer provider
    and replaces them with mapped exporters that convert Pipecat attributes
    to Future AGI conventions.

    Args:
        transport: The transport protocol to match existing exporters (Transport enum)

    Returns:
        True if at least one exporter was swapped; False otherwise.
    """
    provider = trace_api.get_tracer_provider()
    swapped_any = False

    active = getattr(provider, "_active_span_processor", None)
    if not active:
        logger.warning("No active span processor found")
        return False

    processors = getattr(active, "_span_processors", tuple())
    for proc in processors:
        exporter = getattr(proc, "span_exporter", None)
        if exporter is None:
            continue

        try:
            if transport == Transport.HTTP:
                from fi_instrumentation.otel import (
                    HTTPSpanExporter as _FIHTTPSpanExporter,
                )

                if isinstance(exporter, _FIHTTPSpanExporter):
                    endpoint = getattr(exporter, "_endpoint", None)
                    headers = getattr(exporter, "_headers", None)
                    new_exporter = MappedHTTPSpanExporter(
                        endpoint=endpoint, headers=headers
                    )

                    if hasattr(proc, "_batch_processor"):
                        setattr(proc._batch_processor, "_exporter", new_exporter)
                    else:
                        setattr(proc, "span_exporter", new_exporter)


"""
Basic Pinecone query example with tracing.
"""

import os
from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from traceai_pinecone import PineconeInstrumentor


def main():
    # Set up environment variables
    os.environ.setdefault("FI_API_KEY", "<your-api-key>")
    os.environ.setdefault("FI_SECRET_KEY", "<your-secret-key>")

    # Register tracer provider
    trace_provider = register(
        project_type=ProjectType.OBSERVE,

"""
RAG (Retrieval-Augmented Generation) pipeline example with full tracing.

This example shows how to trace both the retrieval (Pinecone) and
generation (OpenAI) steps of a RAG pipeline.
"""

import os
from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from traceai_pinecone import PineconeInstrumentor
from traceai_openai import OpenAIInstrumentor


def setup_tracing():
    """Set up tracing for the RAG pipeline."""
    os.environ.setdefault("FI_API_KEY", "<your-api-key>")
    os.environ.setdefault("FI_SECRET_KEY", "<your-secret-key>")

    trace_provider = register(
        project_type=ProjectType.OBSERVE,
        project_name="rag-pipeline"
    )

    # Instrument both Pinecone and OpenAI
    PineconeInstrumentor().instrument(tracer_provider=trace_provider)
    OpenAIInstrumentor().instrument(tracer_provider=trace_provider)

    return trace_provider


def rag_query(question: str, index, client) -> str:
    """
    Execute a RAG query with full tracing.

    Args:
        question: User question
        index: Pinecone index
        client: OpenAI client

    Returns:
        Generated answer
    """
    # Step 1: Generate embedding for the question
    embedding_response = client.embeddings.create(

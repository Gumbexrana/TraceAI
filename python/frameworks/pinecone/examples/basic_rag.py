"""
Basic RAG Example with Pinecone and OpenAI

This example demonstrates how to build a simple RAG pipeline with:
1. Pinecone vector database instrumentation
2. OpenAI for embeddings and completions

Prerequisites:
    pip install traceai-pinecone traceai-openai pinecone-client openai

Environment variables:
    PINECONE_API_KEY: Your Pinecone API key
    OPENAI_API_KEY: Your OpenAI API key
    FI_API_KEY: Your Future AGI API key (optional)
    FI_SECRET_KEY: Your Future AGI secret key (optional)
"""

import os
from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from traceai_pinecone import PineconeInstrumentor
from traceai_openai import OpenAIInstrumentor
import pinecone
import openai

# Initialize tracing
trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name="pinecone-rag-example"
)

# Instrument both Pinecone and OpenAI
PineconeInstrumentor().instrument(tracer_provider=trace_provider)
OpenAIInstrumentor().instrument(tracer_provider=trace_provider)

# Initialize clients
pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
openai_client = openai.OpenAI()

# Configuration
INDEX_NAME = "rag-demo"
EMBEDDING_MODEL = "text-embedding-3-small"
COMPLETION_MODEL = "gpt-4o-mini"


def get_embedding(text: str) -> list[float]:
    """Generate embedding for text using OpenAI."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def index_documents(documents: list[dict]):
    """Index documents in Pinecone."""
    index = pc.Index(INDEX_NAME)

    vectors = []
    for i, doc in enumerate(documents):
        embedding = get_embedding(doc["content"])
        vectors.append({
            "id": f"doc_{i}",
            "values": embedding,
            "metadata": {
                "title": doc.get("title", ""),
                "content": doc["content"][:1000],  # Store first 1000 chars
            }
        })


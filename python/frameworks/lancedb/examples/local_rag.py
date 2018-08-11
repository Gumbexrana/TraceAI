"""
Local RAG Example with LanceDB

LanceDB is an embedded vector database that runs locally without a server.
Perfect for development, edge deployments, and serverless applications.

Prerequisites:
    pip install traceai-lancedb lancedb

Environment variables:
    FI_API_KEY: Your Future AGI API key (optional)
"""

import lancedb
from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from traceai_lancedb import LanceDBInstrumentor

# Initialize tracing
trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name="lancedb-local-rag"
)

# Instrument LanceDB
LanceDBInstrumentor().instrument(tracer_provider=trace_provider)

# Connect to local database (creates ~/.lancedb by default)
db = lancedb.connect("~/.lancedb/demo")

TABLE_NAME = "documents"
DIMENSION = 384


def generate_fake_embedding(text: str) -> list[float]:
    """Generate fake embedding for demo."""
    import hashlib
    hash_bytes = hashlib.sha256(text.encode()).digest()
    embedding = []
    for i in range(DIMENSION):
        byte_val = hash_bytes[i % len(hash_bytes)]
        embedding.append((byte_val / 255.0) * 2 - 1)
    return embedding


def create_table_with_data(documents: list[dict]):
    """Create table and add initial data."""
    data = []
    for i, doc in enumerate(documents):
        data.append({
            "id": i,
            "text": doc["text"],
            "category": doc.get("category", "general"),
            "vector": generate_fake_embedding(doc["text"]),
        })

    # Create table (traced)
    table = db.create_table(TABLE_NAME, data, mode="overwrite")
    print(f"Created table with {len(data)} documents")
    return table


def add_documents(documents: list[dict]):
    """Add more documents to existing table."""
    table = db.open_table(TABLE_NAME)

    data = []
    for i, doc in enumerate(documents):
        data.append({
            "id": 100 + i,
            "text": doc["text"],
            "category": doc.get("category", "general"),
            "vector": generate_fake_embedding(doc["text"]),
        })

    # Add (traced)
    table.add(data)
    print(f"Added {len(data)} documents")


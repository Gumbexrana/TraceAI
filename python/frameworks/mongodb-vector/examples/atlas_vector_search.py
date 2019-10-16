"""
MongoDB Atlas Vector Search Example

This example demonstrates vector search using MongoDB Atlas with tracing.

Prerequisites:
    pip install traceai-mongodb pymongo

Environment variables:
    MONGODB_URI: Your MongoDB Atlas connection string
    FI_API_KEY: Your Future AGI API key (optional)

Note: Requires MongoDB Atlas cluster with Vector Search index configured.
"""

import os
from pymongo import MongoClient
from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from traceai_mongodb import MongoDBInstrumentor

# Initialize tracing
trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name="mongodb-atlas-vector-search"
)

# Instrument MongoDB
MongoDBInstrumentor().instrument(tracer_provider=trace_provider)

# Connect to MongoDB Atlas
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
client = MongoClient(MONGODB_URI)

DB_NAME = "vector_demo"
COLLECTION_NAME = "documents"
DIMENSION = 384

db = client[DB_NAME]
collection = db[COLLECTION_NAME]


def generate_fake_embedding(text: str) -> list[float]:
    """Generate fake embedding for demo."""
    import hashlib
    hash_bytes = hashlib.sha256(text.encode()).digest()
    embedding = []
    for i in range(DIMENSION):
        byte_val = hash_bytes[i % len(hash_bytes)]
        embedding.append((byte_val / 255.0) * 2 - 1)
    return embedding


def add_documents(documents: list[dict]):
    """Add documents with embeddings."""
    docs_to_insert = []
    for doc in documents:
        docs_to_insert.append({
            "text": doc["text"],
            "category": doc.get("category", "general"),
            "embedding": generate_fake_embedding(doc["text"]),
        })

    # Insert many (traced)
    result = collection.insert_many(docs_to_insert)
    print(f"Inserted {len(result.inserted_ids)} documents")


def vector_search(query: str, limit: int = 5, num_candidates: int = 100) -> list[dict]:
    """
    Perform vector search using MongoDB Atlas $vectorSearch.

    Note: Requires a vector search index named 'vector_index' on the 'embedding' field.
    """
    query_vector = generate_fake_embedding(query)

    # Vector search aggregation (traced)
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": num_candidates,
                "limit": limit,
            }
        },
        {

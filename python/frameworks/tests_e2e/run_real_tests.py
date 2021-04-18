#!/usr/bin/env python3
"""
Real E2E Integration Tests for Vector Database Instrumentation

This script runs actual tests with real data against embedded vector databases.
No mocking - everything is real.
"""

import sys
import os
import tempfile
import shutil

# Add parent paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'chromadb'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lancedb'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'qdrant'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'weaviate'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'milvus'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mongodb-vector'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pgvector'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'redis-vector'))

# Docker service ports (can be overridden by environment variables)
MONGODB_PORT = int(os.environ.get("MONGODB_PORT", "37017"))
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", "25432"))
REDIS_PORT = int(os.environ.get("REDIS_PORT", "26379"))

from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

# Global tracer setup - only once
EXPORTER = InMemorySpanExporter()
RESOURCE = Resource(attributes={"service.name": "real-e2e-test"})
PROVIDER = trace_sdk.TracerProvider(resource=RESOURCE)
PROVIDER.add_span_processor(SimpleSpanProcessor(EXPORTER))
trace_api.set_tracer_provider(PROVIDER)


def generate_embedding(text: str, dim: int = 384) -> list[float]:
    """Generate deterministic embedding from text."""
    import hashlib
    hash_bytes = hashlib.sha256(text.encode()).digest()
    embedding = []
    for i in range(dim):
        byte_val = hash_bytes[i % len(hash_bytes)]
        embedding.append((byte_val / 255.0) * 2 - 1)
    return embedding


def print_spans(title: str):
    """Print captured spans."""
    spans = EXPORTER.get_finished_spans()
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Total spans captured: {len(spans)}")
    print()
    for span in spans:
        attrs = dict(span.attributes or {})
        status = "✓" if span.status.is_ok else "✗"
        print(f"  {status} Span: {span.name}")
        print(f"      db.system: {attrs.get('db.system', 'N/A')}")
        print(f"      db.operation.name: {attrs.get('db.operation.name', 'N/A')}")
        if 'db.vector.query.top_k' in attrs:
            print(f"      db.vector.query.top_k: {attrs['db.vector.query.top_k']}")
        if 'db.vector.results.count' in attrs:
            print(f"      db.vector.results.count: {attrs['db.vector.results.count']}")
        if 'db.vector.upsert.count' in attrs:
            print(f"      db.vector.upsert.count: {attrs['db.vector.upsert.count']}")
        print()
    span_count = len(spans)
    EXPORTER.clear()
    return span_count


def test_chromadb_real():
    """Test ChromaDB with real data."""
    print("\n" + "="*70)
    print("  TESTING: ChromaDB (Embedded)")
    print("="*70)

    # Import and instrument
    from traceai_chromadb import ChromaDBInstrumentor
    import chromadb

    instrumentor = ChromaDBInstrumentor()
    instrumentor.instrument(tracer_provider=trace_api.get_tracer_provider())

    try:
        # Create ephemeral client
        client = chromadb.Client()
        print("  ✓ Created ChromaDB client")

        # Create collection
        collection = client.create_collection(
            name="test_collection",
            metadata={"hnsw:space": "cosine"}
        )
        print("  ✓ Created collection 'test_collection'")

        # Add documents with embeddings
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing analyzes human language.",
            "Computer vision enables machines to interpret images.",
            "Reinforcement learning uses reward-based training.",
        ]

        embeddings = [generate_embedding(doc) for doc in documents]
        ids = [f"doc_{i}" for i in range(len(documents))]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=[{"category": "ml"} for _ in documents],
        )
        print(f"  ✓ Added {len(documents)} documents with embeddings")

        # Query
        query_embedding = generate_embedding("neural networks and AI")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "distances", "metadatas"],
        )
        print(f"  ✓ Query returned {len(results['ids'][0])} results")

        # Get by ID
        get_results = collection.get(ids=["doc_0", "doc_1"])
        print(f"  ✓ Get returned {len(get_results['ids'])} documents")

        # Count
        count = collection.count()
        print(f"  ✓ Collection count: {count}")

        # Delete
        collection.delete(ids=["doc_4"])
        print("  ✓ Deleted 1 document")

        # Verify count after delete
        new_count = collection.count()
        print(f"  ✓ New count after delete: {new_count}")

        # Print spans
        span_count = print_spans("ChromaDB Spans")

        # Cleanup
        client.delete_collection("test_collection")

        print(f"\n  ✓ ChromaDB TEST PASSED - {span_count} spans captured")
        return True, span_count

    except Exception as e:
        print(f"\n  ✗ ChromaDB TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, 0
    finally:
        instrumentor.uninstrument()


def test_lancedb_real():
    """Test LanceDB with real data."""
    print("\n" + "="*70)
    print("  TESTING: LanceDB (Embedded)")
    print("="*70)

    # Import and instrument
    from traceai_lancedb import LanceDBInstrumentor
    import lancedb

    instrumentor = LanceDBInstrumentor()
    instrumentor.instrument(tracer_provider=trace_api.get_tracer_provider())

    # Create temp directory for LanceDB
    temp_dir = tempfile.mkdtemp()

    try:
        # Connect to database
        db = lancedb.connect(temp_dir)
        print(f"  ✓ Connected to LanceDB at {temp_dir}")

        # Create table with data
        documents = [
            "Machine learning algorithms learn patterns from data.",
            "Deep neural networks have multiple hidden layers.",
            "Natural language processing understands human text.",
            "Computer vision processes images and videos.",
            "Reinforcement learning optimizes through rewards.",
        ]

        data = [
            {
                "id": i,
                "text": doc,
                "vector": generate_embedding(doc),
            }
            for i, doc in enumerate(documents)
        ]

        table = db.create_table("documents", data)
        print(f"  ✓ Created table with {len(documents)} documents")

        # Search
        query_vector = generate_embedding("neural networks")
        results = table.search(query_vector).limit(3).to_list()
        print(f"  ✓ Search returned {len(results)} results")

        # Add more data
        new_data = [
            {"id": 10, "text": "Transformers revolutionized NLP.", "vector": generate_embedding("Transformers")}
        ]
        table.add(new_data)
        print("  ✓ Added 1 new document")

        # Search again
        results2 = table.search(query_vector).limit(5).to_list()
        print(f"  ✓ Second search returned {len(results2)} results")

        # Delete
        table.delete("id = 0")
        print("  ✓ Deleted document with id=0")

        # Print spans
        span_count = print_spans("LanceDB Spans")

        print(f"\n  ✓ LanceDB TEST PASSED - {span_count} spans captured")
        return True, span_count

    except Exception as e:
        print(f"\n  ✗ LanceDB TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, 0
    finally:
        instrumentor.uninstrument()
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_qdrant_real():
    """Test Qdrant with real data (in-memory mode)."""
    print("\n" + "="*70)
    print("  TESTING: Qdrant (In-Memory)")
    print("="*70)

    # Import and instrument
    from traceai_qdrant import QdrantInstrumentor
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    instrumentor = QdrantInstrumentor()
    instrumentor.instrument(tracer_provider=trace_api.get_tracer_provider())

    try:
        # Create in-memory client
        client = QdrantClient(":memory:")
        print("  ✓ Created Qdrant in-memory client")

        collection_name = "test_collection"

        # Create collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print(f"  ✓ Created collection '{collection_name}'")

        # Add points
        documents = [
            "Machine learning enables pattern recognition.",
            "Deep learning uses multi-layer neural networks.",
            "NLP processes human language data.",
            "Computer vision interprets visual information.",
            "RL agents learn through trial and error.",
        ]

        points = [
            PointStruct(
                id=i,
                vector=generate_embedding(doc),
                payload={"text": doc, "category": "ml"}
            )
            for i, doc in enumerate(documents)
        ]

        client.upsert(collection_name=collection_name, points=points)
        print(f"  ✓ Upserted {len(points)} points")

        # Query (using query_points - new Qdrant API)
        query_vector = generate_embedding("neural networks AI")
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=3,
            with_payload=True,
        )
        print(f"  ✓ Query returned {len(results.points)} results")

        # Retrieve by ID
        retrieved = client.retrieve(
            collection_name=collection_name,
            ids=[0, 1],
            with_payload=True,
        )
        print(f"  ✓ Retrieved {len(retrieved)} points by ID")

        # Count
        count = client.count(collection_name=collection_name)
        print(f"  ✓ Collection count: {count.count}")

        # Delete
        client.delete(
            collection_name=collection_name,
            points_selector=[4],
        )
        print("  ✓ Deleted 1 point")

        # Scroll
        scroll_results = client.scroll(
            collection_name=collection_name,
            limit=10,
            with_payload=True,
        )
        print(f"  ✓ Scroll returned {len(scroll_results[0])} points")

        # Print spans
        span_count = print_spans("Qdrant Spans")

        print(f"\n  ✓ Qdrant TEST PASSED - {span_count} spans captured")
        return True, span_count

    except Exception as e:
        print(f"\n  ✗ Qdrant TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, 0
    finally:
        instrumentor.uninstrument()


def test_weaviate_real():
    """Test Weaviate with real data (embedded mode)."""
    print("\n" + "="*70)
    print("  TESTING: Weaviate (Embedded)")
    print("="*70)

    # Import and instrument
    from traceai_weaviate import WeaviateInstrumentor
    import weaviate
    from weaviate.classes.config import Property, DataType, Configure

    instrumentor = WeaviateInstrumentor()
    instrumentor.instrument(tracer_provider=trace_api.get_tracer_provider())

    client = None
    try:
        # Create embedded client
        client = weaviate.connect_to_embedded()
        print("  ✓ Created Weaviate embedded client")

        collection_name = "TestDocuments"

        # Delete collection if it exists
        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)

        # Create collection with vector index
        collection = client.collections.create(
            name=collection_name,
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="category", data_type=DataType.TEXT),
            ],
            vectorizer_config=Configure.Vectorizer.none(),
        )
        print(f"  ✓ Created collection '{collection_name}'")

        # Insert documents with embeddings
        documents = [
            "Machine learning enables pattern recognition.",

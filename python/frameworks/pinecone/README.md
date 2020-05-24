# traceAI-pinecone

OpenTelemetry instrumentation for [Pinecone](https://www.pinecone.io/) vector database.

## Installation

```bash
pip install traceAI-pinecone
```

## Quick Start

```python
import os
from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from traceai_pinecone import PineconeInstrumentor
import pinecone

# Set up environment
os.environ["FI_API_KEY"] = "<your-api-key>"
os.environ["FI_SECRET_KEY"] = "<your-secret-key>"

# Register tracer
trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name="my-rag-app"
)

# Instrument Pinecone
PineconeInstrumentor().instrument(tracer_provider=trace_provider)

# Use Pinecone as normal - all operations are traced!
pc = pinecone.Pinecone(api_key="your-pinecone-api-key")
index = pc.Index("my-index")

# Query vectors
results = index.query(
    vector=[0.1] * 1536,
    top_k=10,
    include_metadata=True
)

# Upsert vectors
index.upsert(
    vectors=[
        {"id": "vec1", "values": [0.1] * 1536, "metadata": {"title": "Doc 1"}},
        {"id": "vec2", "values": [0.2] * 1536, "metadata": {"title": "Doc 2"}},
    ]
)
```

## Instrumented Operations

| Operation | Span Name | Description |
|-----------|-----------|-------------|
| `query` | `pinecone query` | Vector similarity search |
| `upsert` | `pinecone upsert` | Insert or update vectors |
| `delete` | `pinecone delete` | Delete vectors by ID or filter |
| `fetch` | `pinecone fetch` | Retrieve vectors by ID |
| `update` | `pinecone update` | Update vector metadata |
| `describe_index_stats` | `pinecone describe_index_stats` | Get index statistics |

## Span Attributes

### Common Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `db.system` | string | Always `"pinecone"` |
| `db.operation.name` | string | Operation name (query, upsert, etc.) |
| `db.namespace` | string | Index name |
| `db.vector.namespace` | string | Pinecone namespace |
| `db.vector.index.name` | string | Index name |
| `db.vector.index.host` | string | Pinecone host URL |

### Query Attributes


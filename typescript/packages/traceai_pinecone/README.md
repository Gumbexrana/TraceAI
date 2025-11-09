# @traceai/pinecone

OpenTelemetry instrumentation for [Pinecone](https://www.pinecone.io/) - the serverless vector database for high-performance AI applications.

## Overview

This package provides automatic tracing for Pinecone operations in Node.js applications, enabling full observability of your vector database interactions in production AI systems, RAG pipelines, and semantic search applications.

## Installation

```bash
npm install @traceai/pinecone
# or
yarn add @traceai/pinecone
# or
pnpm add @traceai/pinecone
```

## Quick Start

### Basic Setup

```typescript
import { PineconeInstrumentation } from "@traceai/pinecone";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { registerInstrumentations } from "@opentelemetry/instrumentation";

// Set up OpenTelemetry
const provider = new NodeTracerProvider();
provider.addSpanProcessor(
  new SimpleSpanProcessor(
    new OTLPTraceExporter({
      url: "https://api.futureagi.com/v1/traces",
      headers: {
        "Authorization": `Bearer ${process.env.FI_API_KEY}`,
        "FI-Project-Name": process.env.FI_PROJECT_NAME,
      },
    })
  )
);
provider.register();

// Register Pinecone instrumentation
registerInstrumentations({
  instrumentations: [
    new PineconeInstrumentation(),
  ],
});

// Now use Pinecone as normal - all operations are traced
import { Pinecone } from "@pinecone-database/pinecone";

const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pinecone.index("my-index");

// This operation will be automatically traced
const results = await index.query({
  vector: [0.1, 0.2, 0.3, /* ... */],
  topK: 10,
  includeMetadata: true,
});
```

### Manual Instrumentation

```typescript
import { PineconeInstrumentation } from "@traceai/pinecone";
import * as pinecone from "@pinecone-database/pinecone";

const instrumentation = new PineconeInstrumentation();
instrumentation.manuallyInstrument(pinecone);
```

## Configuration Options

```typescript
interface PineconeInstrumentationConfig {
  // Whether to capture query vectors in spans (may be large)
  captureQueryVectors?: boolean;

  // Whether to capture result vectors in spans (may be large)
  captureResultVectors?: boolean;
}

// Example with all options
const instrumentation = new PineconeInstrumentation({
  instrumentationConfig: {
    captureQueryVectors: false,  // Recommended for production
    captureResultVectors: false,
  },
  traceConfig: {
    maskInputs: false,
    maskOutputs: false,
  },
});
```

## Traced Operations

| Operation | Span Name | Description |
|-----------|-----------|-------------|
| `query` | `pinecone query` | Semantic similarity search |
| `upsert` | `pinecone upsert` | Insert or update vectors |
| `fetch` | `pinecone fetch` | Retrieve vectors by ID |
| `update` | `pinecone update` | Update vector metadata |
| `deleteOne` | `pinecone delete` | Delete single vector |
| `deleteMany` | `pinecone delete_many` | Delete multiple vectors |
| `deleteAll` | `pinecone delete_all` | Delete all vectors in namespace |
| `listPaginated` | `pinecone list` | List vectors with pagination |
| `describeIndexStats` | `pinecone describe_stats` | Get index statistics |

## Span Attributes

Each span includes these semantic convention attributes:

| Attribute | Description | Example |
|-----------|-------------|---------|
| `db.system` | Database system identifier | `pinecone` |
| `db.operation.name` | Operation being performed | `query`, `upsert` |
| `db.vector.index.name` | Index name | `my-index` |
| `db.vector.namespace` | Namespace (if used) | `documents` |
| `db.vector.query.top_k` | Number of results requested | `10` |
| `db.vector.query.filter` | Query filter (JSON) | `{"category": "docs"}` |
| `db.vector.query.include_metadata` | Whether metadata included | `true` |
| `db.vector.query.include_vectors` | Whether vectors included | `false` |
| `db.vector.results.count` | Number of results returned | `10` |
| `db.vector.upsert.count` | Number of vectors upserted | `100` |
| `db.vector.upsert.dimensions` | Vector dimensions | `1536` |
| `db.vector.delete.count` | Number of vectors deleted | `50` |
| `db.vector.delete.all` | Whether all deleted | `true` |
| `db.vector.index.dimensions` | Index dimensions | `1536` |
| `fi.span.kind` | TraceAI span kind | `VECTOR_DB` |

## Real-World Use Cases

### 1. Production RAG with Serverless Pinecone

High-scale retrieval-augmented generation:

```typescript
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const pinecone = new Pinecone();
const openai = new OpenAI();

const index = pinecone.index("knowledge-base");

// Ingest documents with namespace isolation
async function ingestDocuments(documents: Document[], namespace: string) {
  const batchSize = 100;

  for (let i = 0; i < documents.length; i += batchSize) {
    const batch = documents.slice(i, i + batchSize);

    // Generate embeddings
    const embeddings = await openai.embeddings.create({
      model: "text-embedding-3-large",
      input: batch.map(d => d.content),
      dimensions: 1536,
    });

    // Upsert to Pinecone (traced automatically)
    await index.namespace(namespace).upsert(
      batch.map((doc, j) => ({
        id: doc.id,
        values: embeddings.data[j].embedding,
        metadata: {
          title: doc.title,
          source: doc.source,
          created_at: doc.createdAt,
          chunk_index: doc.chunkIndex,
        },
      }))
    );
  }
}

// RAG query with hybrid search (traced automatically)
async function ragQuery(question: string, namespace: string, filters?: object) {
  // Generate query embedding
  const queryEmbedding = await openai.embeddings.create({
    model: "text-embedding-3-large",
    input: question,
    dimensions: 1536,
  });

  // Semantic search with metadata filtering
  const results = await index.namespace(namespace).query({
    vector: queryEmbedding.data[0].embedding,
    topK: 5,
    includeMetadata: true,
    filter: filters,
  });

  // Build context from results
  const context = results.matches
    .map(m => m.metadata?.content)
    .filter(Boolean)
    .join("\n\n");

  // Generate response
  const response = await openai.chat.completions.create({
    model: "gpt-4-turbo",
    messages: [
      {
        role: "system",
        content: `Answer based on this context:\n${context}\n\nIf the context doesn't contain relevant information, say so.`,
      },
      { role: "user", content: question },
    ],
  });

  return {
    answer: response.choices[0].message.content,
    sources: results.matches.map(m => ({
      id: m.id,
      score: m.score,
      title: m.metadata?.title,
    })),
  };
}
```

### 2. Multi-Tenant SaaS Application

Isolate customer data using namespaces:

```typescript
class MultiTenantVectorStore {
  private index: any;

  constructor(indexName: string) {
    const pinecone = new Pinecone();
    this.index = pinecone.index(indexName);
  }

  // Store customer data in isolated namespace (traced)
  async storeForTenant(tenantId: string, documents: any[]) {
    const ns = this.index.namespace(`tenant_${tenantId}`);

    await ns.upsert(documents.map(doc => ({
      id: `${tenantId}_${doc.id}`,
      values: doc.embedding,
      metadata: {
        tenant_id: tenantId,
        ...doc.metadata,
      },
    })));
  }

  // Query within tenant namespace only (traced)
  async queryForTenant(tenantId: string, queryVector: number[], options: any = {}) {

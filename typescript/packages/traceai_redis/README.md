# @traceai/redis

OpenTelemetry instrumentation for [Redis Vector Search](https://redis.io/docs/stack/search/reference/vectors/) (RediSearch) in Node.js/TypeScript applications.

## Installation

```bash
npm install @traceai/redis
# or
pnpm add @traceai/redis
# or
yarn add @traceai/redis
```

## Prerequisites

- Node.js >= 18
- Redis client (`redis` >= 4.0.0)
- Redis Stack or Redis with RediSearch module
- OpenTelemetry SDK configured in your application

## Quick Start

```typescript
import { RedisInstrumentation } from "@traceai/redis";
import { createClient } from "redis";

// Initialize instrumentation
const instrumentation = new RedisInstrumentation({
  traceConfig: {
    maskInputs: false,
    maskOutputs: false,
  },
});

// Enable instrumentation
instrumentation.enable();

// Manually instrument the redis module
import * as redis from "redis";
instrumentation.manuallyInstrument(redis);

// Now all Redis operations will be traced
const client = createClient({ url: process.env.REDIS_URL });
await client.connect();

// Vector search with FT.SEARCH (traced)
const results = await client.ft.search("idx:products", "*=>[KNN 10 @embedding $vector AS score]", {
  PARAMS: {
    vector: Buffer.from(new Float32Array([0.1, 0.2, 0.3]).buffer),
  },
  RETURN: ["name", "price", "score"],
  SORTBY: { BY: "score" },
  DIALECT: 2,
});
```

## Configuration Options

```typescript
interface RedisInstrumentationConfig {
  // Enable/disable the instrumentation
  enabled?: boolean;

  // Capture query vectors in span attributes
  captureQueryVectors?: boolean;

  // Capture result documents
  captureDocuments?: boolean;
}

interface TraceConfigOptions {
  // Mask sensitive input data
  maskInputs?: boolean;

  // Mask sensitive output data
  maskOutputs?: boolean;
}
```

## Traced Operations

The instrumentation automatically traces Redis operations with special handling for vector search:

### Vector Search Operations

- `FT.SEARCH` - Vector similarity search
- `FT.AGGREGATE` - Aggregation with vector scoring
- `FT.CREATE` - Create search index

### Hash Operations (for vector storage)

- `HSET` - Store hash with vector field
- `HGET` / `HMGET` - Retrieve hash fields
- `HDEL` - Delete hash

### JSON Operations (for document storage)

- `JSON.SET` - Store JSON document
- `JSON.GET` - Retrieve JSON document

## Span Attributes

Each traced operation includes relevant attributes:

| Attribute                 | Description                                    |
| ------------------------- | ---------------------------------------------- |
| `db.system`               | Always "redis"                                 |
| `db.operation`            | Operation name (e.g., "FT.SEARCH", "HSET")     |
| `db.redis.index_name`     | Search index name                              |
| `db.redis.query`          | Search query                                   |
| `db.redis.vector_field`   | Vector field name                              |
| `db.redis.k`              | Number of neighbors (KNN)                      |
| `db.redis.result_count`   | Number of results returned                     |

## Real-World Use Cases

### 1. E-commerce Product Search

```typescript
import { RedisInstrumentation } from "@traceai/redis";
import { createClient, SchemaFieldTypes, VectorAlgorithms } from "redis";

const instrumentation = new RedisInstrumentation();
instrumentation.enable();

const client = createClient({ url: process.env.REDIS_URL });
await client.connect();

// Create product search index
async function createProductIndex() {
  try {
    await client.ft.create(
      "idx:products",
      {
        "$.name": { type: SchemaFieldTypes.TEXT, AS: "name" },
        "$.description": { type: SchemaFieldTypes.TEXT, AS: "description" },
        "$.category": { type: SchemaFieldTypes.TAG, AS: "category" },
        "$.price": { type: SchemaFieldTypes.NUMERIC, AS: "price" },
        "$.embedding": {
          type: SchemaFieldTypes.VECTOR,
          AS: "embedding",
          ALGORITHM: VectorAlgorithms.HNSW,
          TYPE: "FLOAT32",
          DIM: 384,
          DISTANCE_METRIC: "COSINE",
        },
      },
      { ON: "JSON", PREFIX: "product:" }
    );
  } catch (e: any) {
    if (!e.message.includes("Index already exists")) throw e;
  }
}

// Semantic product search (traced)
async function searchProducts(queryVector: number[], filters?: ProductFilters) {
  const query = buildSearchQuery(filters);

  const results = await client.ft.search(
    "idx:products",
    `${query}=>[KNN 20 @embedding $vector AS score]`,
    {
      PARAMS: {
        vector: Buffer.from(new Float32Array(queryVector).buffer),
      },
      RETURN: ["name", "description", "price", "category", "score"],
      SORTBY: { BY: "score" },
      LIMIT: { from: 0, size: 20 },
      DIALECT: 2,
    }
  );

  return results.documents.map((doc) => ({
    id: doc.id.replace("product:", ""),
    ...doc.value,
  }));
}

function buildSearchQuery(filters?: ProductFilters): string {
  const parts: string[] = ["*"];

  if (filters?.category) {
    parts.push(`@category:{${filters.category}}`);
  }
  if (filters?.minPrice !== undefined) {
    parts.push(`@price:[${filters.minPrice} +inf]`);
  }

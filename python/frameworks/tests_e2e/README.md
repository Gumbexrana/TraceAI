# End-to-End Integration Tests for Vector Database Instrumentation

This directory contains E2E integration tests that verify the instrumentation
works correctly with real database instances.

## Quick Start

### For Embedded Databases (No Setup Required)

ChromaDB and LanceDB run in-memory and don't require external setup:

```bash
# Install dependencies
pip install chromadb lancedb

# Run tests
pytest tests_e2e/test_chromadb_e2e.py -v
pytest tests_e2e/test_lancedb_e2e.py -v
```

### For External Databases

1. **Start the databases using Docker Compose:**

```bash
cd tests_e2e
docker-compose up -d
```

2. **Wait for databases to be ready** (about 30 seconds)

3. **Run the integration tests:**

```bash
# Run all integration tests
pytest tests_e2e/ -v -m integration

# Run specific database tests
pytest tests_e2e/test_qdrant_e2e.py -v
pytest tests_e2e/test_milvus_e2e.py -v
pytest tests_e2e/test_pgvector_e2e.py -v
pytest tests_e2e/test_redis_e2e.py -v
```

4. **Stop the databases:**

```bash
docker-compose down
```

## Database Connection Details

| Database | Port | Connection String |
|----------|------|-------------------|
| Qdrant | 6333 | `http://localhost:6333` |
| Weaviate | 8080 | `http://localhost:8080` |
| Milvus | 19530 | `localhost:19530` |
| PostgreSQL | 5432 | `postgresql://postgres:postgres@localhost:5432/vectordb` |
| Redis | 6379 | `redis://localhost:6379` |

## Test Markers

Tests are marked with database-specific markers for selective execution:

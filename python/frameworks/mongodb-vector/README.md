# traceAI-mongodb

OpenTelemetry instrumentation for [MongoDB Atlas Vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/).

## Installation

```bash
pip install traceAI-mongodb
```

## Quick Start

```python
from fi_instrumentation import register
from traceai_mongodb import MongoDBInstrumentor
from pymongo import MongoClient

trace_provider = register(project_name="my-rag-app")
MongoDBInstrumentor().instrument(tracer_provider=trace_provider)

client = MongoClient("mongodb+srv://...")
db = client["mydb"]
collection = db["documents"]

# Vector search operations are traced

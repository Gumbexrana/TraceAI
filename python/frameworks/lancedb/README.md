# traceAI-lancedb

OpenTelemetry instrumentation for [LanceDB](https://lancedb.com/) embedded vector database.

## Installation

```bash
pip install traceAI-lancedb
```

## Quick Start

```python
from fi_instrumentation import register
from traceai_lancedb import LanceDBInstrumentor
import lancedb


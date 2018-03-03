# Groq OpenTelemetry Integration

## Overview
This integration provides support for using OpenTelemetry with the Groq framework. It enables tracing and monitoring of applications built with Groq.

## Installation

1. **Install traceAI Groq**

```bash
pip install traceAI-groq
```


### Set Environment Variables
Set up your environment variables to authenticate with FutureAGI.

```python
import os

os.environ["FI_API_KEY"] = FI_API_KEY
os.environ["FI_SECRET_KEY"] = FI_SECRET_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
```

## Quickstart

### Register Tracer Provider
Set up the trace provider to establish the observability pipeline. The trace provider:

```python
from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType

trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name="groq_app"
)
```

### Configure Groq Instrumentation
Instrument the Groq client to enable telemetry collection. This step ensures that all interactions with the Groq SDK are tracked and monitored.

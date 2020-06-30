# Pipecat OpenTelemetry Integration

## Overview
This integration provides support for using OpenTelemetry with Pipecat applications. It enables tracing and monitoring of voice applications built with Pipecat, with automatic attribute mapping to Future AGI conventions.

## Installation

1. **Install traceAI Pipecat**

```bash
pip install traceAI-pipecat
```

### Set Environment Variables
Set up your environment variables to authenticate with FutureAGI and Pipecat.

```python
import os

os.environ["FI_API_KEY"] = FI_API_KEY
os.environ["FI_SECRET_KEY"] = FI_SECRET_KEY
```

## Quickstart

This method automatically updates your existing span exporters to include Pipecat attribute mapping.

#### Register Tracer Provider
Set up the trace provider to establish the observability pipeline:

```python
from fi_instrumentation.otel import register, Transport, ProjectType

trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name="Pipecat Voice App",
    set_global_tracer_provider=True,
)
```

#### Install Attribute Mapping
Install attribute mapping to convert Pipecat attributes to Future AGI conventions:

```python
from traceai_pipecat import enable_http_attribute_mapping

# For HTTP transport
success = enable_http_attribute_mapping()

# For gRPC transport
from traceai_pipecat import enable_grpc_attribute_mapping
success = enable_grpc_attribute_mapping()

# Or specify transport explicitly via enum
from traceai_pipecat import enable_fi_attribute_mapping
from fi_instrumentation.otel import Transport
success = enable_fi_attribute_mapping(transport=Transport.HTTP)  # or Transport.GRPC
```

## Features

### Automatic Attribute Mapping
The integration automatically maps Pipecat-specific attributes to Future AGI conventions:

- **LLM Operations**: Maps `gen_ai.system`, `gen_ai.request.model` to `llm.provider`, `llm.model_name`
- **Input/Output**: Maps `input`, `output`, `transcript` to structured Future AGI format
- **Token Usage**: Maps `gen_ai.usage.*` to `llm.token_count.*`
- **Tools**: Maps tool-related attributes to Future AGI tool conventions
- **Session Data**: Maps conversation and session information
- **Metadata**: Consolidates miscellaneous attributes into structured metadata

### Transport Support
- **HTTP**: Full support for HTTP transport with automatic endpoint detection
- **gRPC**: Support for gRPC transport (requires `fi-instrumentation[grpc]`)

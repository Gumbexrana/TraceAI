# traceAI-a2a

**OpenTelemetry instrumentation for the [Google Agent-to-Agent (A2A) Protocol](https://google.github.io/A2A/) — distributed tracing across multi-agent boundaries.**

[![PyPI](https://img.shields.io/pypi/v/traceAI-a2a)](https://pypi.org/project/traceAI-a2a/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)
[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-1.0+-purple)](https://opentelemetry.io/)

---

## The Problem

When an orchestrator agent calls a remote specialist agent over A2A, the trace breaks. The child agent starts a fresh trace. There is no parent-child link, no distributed context propagation, no visibility into what the remote agent did.

`traceai-a2a` fixes this.

---

## What Gets Captured

| Data | Attribute |
|------|-----------|
| Task ID | `gen_ai.a2a.task.id` |
| Task State | `gen_ai.a2a.task.state` |
| Remote Agent URL | `gen_ai.a2a.agent.url` |
| AgentCard Name | `gen_ai.a2a.agent.card.name` |
| AgentCard Version | `gen_ai.a2a.agent.card.version` |
| Message Role | `gen_ai.a2a.message.role` |
| Message Parts Count | `gen_ai.a2a.message.parts.count` |
| SSE Streaming | `gen_ai.a2a.streaming` |
| Artifact Type | `gen_ai.a2a.artifact.type` |
| Propagated Trace ID | `gen_ai.a2a.propagated_trace_id` |

---

## Install

```bash
pip install traceAI-a2a
```

For A2A SDK support:
```bash
pip install "traceAI-a2a[a2a]"
```

---

## Quickstart

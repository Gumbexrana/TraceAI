## [0.2.0] - 2026-07-30
### Fixed
- `LangGraphInstrumentor` no longer breaks async graph nodes. The previous build monkey-patched `StateGraph.add_node`/`compile` and wrapped node functions with a sync wrapper, which returned a coroutine from `async def` nodes and raised `INVALID_GRAPH_NODE_RETURN_VALUE`. It also corrupted per-request state under concurrency and traced HITL `interrupt()` pauses as errors.
- LangGraph node, tool and LLM spans (plus graph-node enrichment via `gen_ai.agent.graph.node_name`/`node_id`, session grouping, and correct HITL interrupt handling) are now captured automatically by `LangChainInstrumentor`'s callback handler — no graph monkey-patching.
- HITL `interrupt()` is kept `OK` (not `ERROR`) for tool/LLM/retriever runs too, not only chain nodes; the graph-node attributes are emitted only on the node's own span.
- Implemented the `on_interrupt`/`on_resume` graph-lifecycle callbacks so langgraph 1.2+ no longer logs an `AttributeError` on every HITL turn.

### Changed
- **BREAKING:** `LangGraphInstrumentor` is now a deprecated no-op shim. It remains importable and `instrument()` is safe to call (it logs a one-time deprecation notice), but tracing is driven entirely by `LangChainInstrumentor`. You can remove the `LangGraphInstrumentor().instrument()` call.
- **BREAKING:** dropped the required `langchain` and `langchain-community` dependencies (the instrumentation only imports `langchain-core`). Pinning `langchain-community` transitively capped `langchain-core < 0.4` and blocked LangChain/LangGraph 1.x; the package now installs cleanly alongside `langchain`/`langgraph` 1.x without `--no-deps`.
- **BREAKING:** minimum Python raised to `>=3.10` (aligns with `langchain-core`/`langgraph` 1.x).

## [0.1.9] - 2025-06-10
### Feature
- Added support for ai-evaluation


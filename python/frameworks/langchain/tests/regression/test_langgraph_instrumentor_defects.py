"""Defects of the monkey-patch LangGraphInstrumentor (customer setup: both on).

Every test here asserts the CORRECT behaviour the customer expects. They fail
today because the LangGraphInstrumentor wraps node functions and compiled-graph
methods, and holds per-request state on a process-wide singleton. Under the
callback-first fix, `LangGraphInstrumentor` becomes a no-op shim, so the wrapping
disappears and these pass.

A concise post-fix invariant several tests rely on: with the shim, no span name
starts with 'langgraph.' — all spans come from the callback path.

Status legend: RED = fails today, flags a finding, must pass after the fix.
"""
from __future__ import annotations

import asyncio

import pytest

from ._helpers import (
    INPUT,
    build_async_graph,
    build_store_graph,
    build_sync_graph,
    span_names,
)


@pytest.mark.asyncio
async def test_async_node_does_not_crash(both_tracing):
    """RED — flags finding #1 (the reported bug).

    `patched_add_node` wraps the async node with the sync `NodeWrapper`, which
    calls it without `await`; LangGraph then rejects the returned coroutine with
    `INVALID_GRAPH_NODE_RETURN_VALUE`. Fails today (raises); passes when the shim
    stops wrapping nodes."""
    app = build_async_graph()
    out = await app.ainvoke(INPUT)  # raises InvalidUpdateError today
    assert out["value"] == 2


def test_node_with_store_injection_does_not_crash(both_tracing):
    """RED — flags finding #3 (functools.wraps leaks the node signature).

    The wrapper copies `__wrapped__`, so LangGraph inspects the original signature,
    sees `store`, and injects it into a wrapper that accepts only `(state, config)`
    → TypeError. Fails today; passes when nodes are no longer wrapped."""
    app = build_store_graph()
    out = app.invoke(INPUT)  # raises TypeError: unexpected keyword argument 'store' today
    assert out["value"] == 2


def test_invoke_emits_no_nested_stream_span(both_tracing):
    """RED — flags finding #5 (invoke -> stream double span).

    The compiled-graph wrappers are instance attributes, so `Pregel.invoke`'s
    internal `self.stream(...)` resolves to the wrapped stream and emits a spurious
    nested `langgraph.stream` span for a plain `invoke()`. Fails today; passes when
    compiled-graph methods are no longer wrapped."""
    app = build_sync_graph()
    app.invoke(INPUT)
    names = span_names(both_tracing)
    assert not any("langgraph.stream" in n for n in names), (
        f"spurious nested stream span emitted for invoke(): {names}"
    )


def test_execution_counter_not_shared_across_conversations(both_tracing):
    """RED — flags finding #2 (shared singleton state across conversations).

    `NodeWrapper._node_execution_counts` is created once per compiled graph and
    never reset, so the 2nd conversation's node span is named
    'langgraph.node.agent[2]'. The '[N]' index is meant to be per-conversation.
    Fails today (a '[2]' span appears); passes when the shim removes node spans."""
    app = build_sync_graph()
    app.invoke(INPUT, {"configurable": {"thread_id": "A"}})
    app.invoke(INPUT, {"configurable": {"thread_id": "B"}})
    names = span_names(both_tracing)
    assert not any("[2]" in n for n in names), (
        f"execution counter leaked across conversations: {names}"
    )



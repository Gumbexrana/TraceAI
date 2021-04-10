"""TraceAI instrumentation for AWS Strands Agents.

This module provides integration between TraceAI and AWS Strands Agents SDK,
enabling comprehensive observability for AI agent workflows.

Strands has built-in OpenTelemetry support, so this integration provides:
1. A convenience function to configure Strands with TraceAI's endpoint
2. Optional callback handlers for extended event capture
3. Helper functions for trace attribute management

Example usage:

    from fi_instrumentation import register
    from fi_instrumentation.fi_types import ProjectType
    from traceai_strands import configure_strands_tracing, StrandsCallbackHandler

    # Setup TraceAI
    trace_provider = register(
        project_type=ProjectType.OBSERVE,
        project_name="strands-agent",
    )


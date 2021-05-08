"""
Basic example of using Together AI instrumentation with traceai-together.

This example demonstrates:
1. Setting up OpenTelemetry tracing
2. Instrumenting the Together AI client
3. Making chat completion requests (both sync and streaming)
4. Using context attributes for session tracking
"""

import together

from fi_instrumentation.otel import register
from fi_instrumentation.instrumentation.context_attributes import using_attributes
from fi_instrumentation.fi_types import (
    EvalName,
    EvalSpanKind,
    EvalTag,
    EvalTagType,
    ProjectType,
)
from traceai_together import TogetherInstrumentor

# Configure trace provider with custom evaluation tags
eval_tags = [
    EvalTag(
        eval_name=EvalName.DETERMINISTIC_EVALS,
        value=EvalSpanKind.LLM,
        type=EvalTagType.OBSERVATION_SPAN,
        custom_eval_name="together_ai_eval",
        mapping={},
    )
]

# Configure trace provider
trace_provider = register(
    project_type=ProjectType.OBSERVE,
    eval_tags=eval_tags,
    project_name="TOGETHER_AI_EXAMPLE",
    project_version_name="v1",
)

# Initialize the Together AI instrumentor
TogetherInstrumentor().instrument(tracer_provider=trace_provider)


def basic_chat_example():
    """Basic chat completion example."""
    client = together.Together()

    response = client.chat.completions.create(
        model="meta-llama/Llama-3-8b-chat-hf",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a haiku about programming."},
        ],
        max_tokens=100,
        temperature=0.7,
    )


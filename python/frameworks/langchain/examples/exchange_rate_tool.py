
import requests
from fi_instrumentation import register
from fi_instrumentation.fi_types import (
    EvalName,
    EvalSpanKind,
    EvalTag,
    EvalTagType,
)
from langchain import agents
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from traceai_langchain import LangChainInstrumentor

# Configure trace provider with custom evaluation tags
eval_tags = [
    EvalTag(
        eval_name=EvalName.DETERMINISTIC_EVALS,
        value=EvalSpanKind.TOOL,
        type=EvalTagType.OBSERVATION_SPAN,
        config={
            "multi_choice": False,
            "choices": ["Yes", "No"],
            "rule_prompt": "Evaluate if the response is correct",
        },
        custom_eval_name="<custom_eval_name>",
    )
]


def setup_instrumentation():
    """Configure and initialize OpenTelemetry instrumentation."""
    trace_provider = register(
        project_name="LANGCHAIN_TEST",
        project_version_name="V4",
        eval_tags=eval_tags,
    )
    LangChainInstrumentor().instrument(tracer_provider=trace_provider)
    return trace_provider

"""
LangGraph Simple Workflow Example with Tracing

Demonstrates LangGraph instrumentation with a simple state machine workflow.
This example shows:
- Graph topology capture
- Node execution tracing
- State transition tracking
- Conditional edge decisions

No external API keys required - uses simple Python functions.
"""

from typing import TypedDict, Literal
from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from langgraph.graph import StateGraph, END
from traceai_langchain import LangChainInstrumentor, LangGraphInstrumentor


# Define the state schema
class WorkflowState(TypedDict):
    """State that flows through the workflow."""
    input: str
    step: int
    results: list
    status: str


# Configure trace provider
trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name="langgraph_simple_example",
    session_name="workflow-demo"
)

# Initialize instrumentors - both LangChain and LangGraph
LangChainInstrumentor().instrument(tracer_provider=trace_provider)
LangGraphInstrumentor().instrument(tracer_provider=trace_provider)


# Define node functions
def process_input(state: WorkflowState) -> dict:
    """First node: Process the input."""
    print(f"Processing input: {state['input']}")
    return {
        "step": 1,
        "results": [f"Processed: {state['input']}"],
        "status": "processing"
    }


def validate_data(state: WorkflowState) -> dict:
    """Second node: Validate the processed data."""
    print(f"Validating data at step {state['step']}")
    is_valid = len(state['input']) > 3
    return {
        "step": 2,
        "results": state["results"] + [f"Validated: {is_valid}"],
        "status": "valid" if is_valid else "invalid"
    }


def transform_data(state: WorkflowState) -> dict:
    """Third node: Transform valid data."""
    print(f"Transforming data at step {state['step']}")
    transformed = state['input'].upper()
    return {
        "step": 3,
        "results": state["results"] + [f"Transformed: {transformed}"],
        "status": "transformed"
    }

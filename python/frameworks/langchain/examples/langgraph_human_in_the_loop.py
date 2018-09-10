"""Human-in-the-Loop (HITL) Example with TraceAI LangGraph Instrumentation.

This example demonstrates TraceAI's comprehensive interrupt/resume tracing,
which is a key competitive advantage over Langfuse (which has this broken).

Key features demonstrated:
1. Intentional interrupts are marked as OK status (not ERROR)
2. Resume spans link back to interrupt spans for unified tracing
3. Human decisions are captured with approver info and feedback
4. Full state preservation across interrupt/resume cycles

Usage (with local test backend):
    # Terminal 1: Start test infrastructure (from core-backend)
    cd /path/to/core-backend
    bin/sdk-test setup
    bin/sdk-test backend

    # Terminal 2: Run this example (from langchain directory)
    export $(cat /path/to/core-backend/.env.test.local | xargs)
    export FI_API_KEY=test_api_key_12345
    export FI_SECRET_KEY=test_secret_key_67890
    export FI_BASE_URL=http://localhost:8001
    python examples/langgraph_human_in_the_loop.py

Usage (with production):
    export OPENAI_API_KEY=your-key
    export FI_API_KEY=your-futureagi-key
    export FI_SECRET_KEY=your-futureagi-secret
    python examples/langgraph_human_in_the_loop.py
"""

import os
from typing import Annotated, TypedDict, Literal
from datetime import datetime

# OpenTelemetry setup
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# TraceAI imports
from traceai_langchain import LangChainInstrumentor, LangGraphInstrumentor


# State definition
class ApprovalState(TypedDict):
    """State for an approval workflow."""
    request_id: str
    request_type: str
    amount: float
    requester: str
    status: str
    approval_notes: str
    approved_by: str
    messages: Annotated[list, lambda x, y: x + y]  # Reducer for messages


def setup_tracing():
    """Set up OpenTelemetry tracing with FutureAGI."""
    # Create tracer provider
    provider = TracerProvider()

    # Determine endpoint based on environment
    # For local testing: FI_BASE_URL=http://localhost:8001
    # For production: uses default FutureAGI endpoint
    base_url = os.environ.get("FI_BASE_URL", "https://trace.futureagi.com")
    endpoint = f"{base_url}/v1/traces"

    # Configure exporter
    exporter = OTLPSpanExporter(
        endpoint=endpoint,
        headers={
            "Authorization": f"Bearer {os.environ.get('FI_API_KEY', '')}",
            "x-secret-key": os.environ.get("FI_SECRET_KEY", ""),
        }
    )

    # Add processor
    provider.add_span_processor(BatchSpanProcessor(exporter))

    # Set as global provider
    trace.set_tracer_provider(provider)

    print(f"[TRACE] Exporting to: {endpoint}")
    return provider


def analyze_request(state: ApprovalState) -> ApprovalState:
    """Analyze the approval request and add risk assessment."""
    messages = [f"[{datetime.now().isoformat()}] Analyzing request {state['request_id']}..."]

    # Simple risk assessment based on amount
    if state["amount"] > 10000:
        risk_level = "HIGH"
        messages.append(f"High value request: ${state['amount']:,.2f} - requires senior approval")
    elif state["amount"] > 1000:
        risk_level = "MEDIUM"
        messages.append(f"Medium value request: ${state['amount']:,.2f} - standard approval process")
    else:
        risk_level = "LOW"
        messages.append(f"Low value request: ${state['amount']:,.2f} - may qualify for auto-approval")

    return {
        "messages": messages,
        "status": f"analyzed_{risk_level.lower()}",
    }


def human_review(state: ApprovalState) -> ApprovalState:
    """Human review checkpoint - this is where we interrupt for HITL.

    In a real application, this would use langgraph's interrupt() function.
    For demo purposes, we simulate the interrupt/resume pattern.
    """
    messages = [f"[{datetime.now().isoformat()}] Awaiting human review for request {state['request_id']}"]

    # In production, you would call:
    # from langgraph.types import interrupt
    # result = interrupt({"request": state, "prompt": "Please approve or reject"})

"""AutoGen v0.4 customer support team example.

This example demonstrates:
- Building a multi-agent customer support system
- Specialized agents for different support tiers
- Tracing support ticket resolution workflow
"""

import asyncio
import os
from typing import Annotated
import json

from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from traceai_autogen import instrument_autogen

# Setup tracing
trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name="autogen-customer-support",
)
instrument_autogen(tracer_provider=trace_provider)

# Import AutoGen v0.4
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient


# Simulated customer database
CUSTOMERS = {
    "C001": {
        "name": "John Doe",
        "plan": "Premium",
        "since": "2022-03-15",
        "tickets_resolved": 12,
    },
    "C002": {
        "name": "Jane Smith",
        "plan": "Basic",
        "since": "2023-08-01",
        "tickets_resolved": 3,
    },
}

# Knowledge base
KB_ARTICLES = {
    "password_reset": "To reset your password: 1) Go to Settings, 2) Click 'Security', 3) Select 'Reset Password', 4) Check your email for the reset link.",
    "billing": "Billing issues can be resolved by: 1) Checking your payment method in Settings > Billing, 2) Updating card information, 3) Contacting support for refunds.",
    "technical": "For technical issues: 1) Clear browser cache, 2) Try incognito mode, 3) Check system status at status.example.com, 4) Contact tech support if issue persists.",
}


def lookup_customer(
    customer_id: Annotated[str, "Customer ID to look up"],
) -> str:
    """Look up customer information."""
    if customer_id in CUSTOMERS:
        return json.dumps(CUSTOMERS[customer_id], indent=2)
    return json.dumps({"error": f"Customer {customer_id} not found"})


def search_knowledge_base(
    topic: Annotated[str, "Topic to search for in knowledge base"],
) -> str:
    """Search the support knowledge base."""
    topic_lower = topic.lower()
    for key, article in KB_ARTICLES.items():
        if key in topic_lower or topic_lower in key:
            return json.dumps({"topic": key, "article": article})

    return json.dumps(
        {
            "message": "No exact match found",
            "available_topics": list(KB_ARTICLES.keys()),
        }
    )


def create_ticket(
    customer_id: Annotated[str, "Customer ID"],
    issue: Annotated[str, "Issue description"],
    priority: Annotated[str, "Priority level: low, medium, high"] = "medium",
) -> str:
    """Create a support ticket."""
    ticket_id = f"TKT-{hash(issue) % 10000:04d}"

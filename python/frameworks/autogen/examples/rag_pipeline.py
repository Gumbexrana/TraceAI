"""AutoGen v0.4 RAG pipeline example with tracing.

This example demonstrates:
- Building a RAG (Retrieval-Augmented Generation) pipeline
- Using tools for document retrieval
- Tracing the full RAG workflow
"""

import asyncio
import os
import json
from typing import Annotated, List

from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from traceai_autogen import instrument_autogen

# Setup tracing
trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name="autogen-rag-pipeline",
)
instrument_autogen(tracer_provider=trace_provider)

# Import AutoGen v0.4
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient


# Simulated document store
DOCUMENTS = {
    "doc1": {
        "title": "Python Best Practices",
        "content": """Python best practices include using virtual environments,
        following PEP 8 style guide, writing docstrings, using type hints,
        handling exceptions properly, and writing unit tests. Always use
        meaningful variable names and keep functions small and focused.""",
        "tags": ["python", "coding", "best-practices"],
    },
    "doc2": {
        "title": "Machine Learning Fundamentals",
        "content": """Machine learning involves training models on data to make
        predictions. Key concepts include supervised learning (classification,
        regression), unsupervised learning (clustering, dimensionality reduction),
        and reinforcement learning. Data preprocessing and feature engineering
        are crucial for model performance.""",
        "tags": ["ml", "ai", "data-science"],
    },
    "doc3": {
        "title": "API Design Guidelines",
        "content": """Good API design follows REST principles: use HTTP methods
        correctly (GET, POST, PUT, DELETE), use meaningful resource names,
        version your API, return appropriate status codes, and provide clear
        error messages. Consider pagination for large collections and use
        consistent naming conventions.""",
        "tags": ["api", "rest", "web-development"],
    },
    "doc4": {
        "title": "Database Optimization",
        "content": """Database optimization includes proper indexing, query
        optimization, connection pooling, and caching strategies. Use EXPLAIN
        to analyze query plans. Consider denormalization for read-heavy workloads
        and use appropriate data types. Regular maintenance like VACUUM and
        ANALYZE keeps performance optimal.""",
        "tags": ["database", "performance", "sql"],
    },
}


def search_documents(
    query: Annotated[str, "Search query to find relevant documents"],
    max_results: Annotated[int, "Maximum number of documents to return"] = 3,
) -> str:

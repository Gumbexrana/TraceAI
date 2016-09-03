"""AutoGen v0.4 code review pipeline example.

This example demonstrates:
- Automated code review workflow with multiple specialized agents
- Static analysis tool integration
- Tracing the complete review pipeline
"""

import asyncio
import os
from typing import Annotated
import json
import re

from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from traceai_autogen import instrument_autogen

# Setup tracing
trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name="autogen-code-review",
)
instrument_autogen(tracer_provider=trace_provider)

# Import AutoGen v0.4
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient


def analyze_code_style(
    code: Annotated[str, "Python code to analyze"],
) -> str:
    """Analyze code for style issues (simulated linter)."""
    issues = []

    lines = code.split("\n")
    for i, line in enumerate(lines, 1):
        # Check line length
        if len(line) > 79:
            issues.append(f"Line {i}: Line too long ({len(line)} > 79 characters)")

        # Check for trailing whitespace
        if line.endswith(" ") or line.endswith("\t"):
            issues.append(f"Line {i}: Trailing whitespace")

        # Check for missing spaces around operators
        if re.search(r"[a-zA-Z0-9][=+\-*/][a-zA-Z0-9]", line):
            issues.append(
                f"Line {i}: Missing spaces around operator"
            )

    return json.dumps(
        {
            "total_issues": len(issues),
            "issues": issues[:10],  # Limit output
            "style_score": max(0, 100 - len(issues) * 5),
        },
        indent=2,
    )


def check_security(
    code: Annotated[str, "Python code to check for security issues"],
) -> str:
    """Check code for common security vulnerabilities."""
    vulnerabilities = []

    # Check for eval usage
    if "eval(" in code:
        vulnerabilities.append(
            {
                "severity": "HIGH",
                "issue": "Use of eval() - potential code injection",
                "recommendation": "Use ast.literal_eval() for safe evaluation",
            }
        )

    # Check for exec usage
    if "exec(" in code:
        vulnerabilities.append(
            {
                "severity": "HIGH",
                "issue": "Use of exec() - potential code injection",
                "recommendation": "Avoid exec() or sanitize input carefully",
            }
        )

    # Check for SQL injection patterns
    if re.search(r'f".*SELECT.*{', code) or re.search(r"f'.*SELECT.*{", code):
        vulnerabilities.append(
            {
                "severity": "CRITICAL",
                "issue": "Potential SQL injection via f-string",
                "recommendation": "Use parameterized queries",
            }
        )


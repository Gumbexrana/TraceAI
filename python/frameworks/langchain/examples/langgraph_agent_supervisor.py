"""
LangGraph Agent Supervisor Example

A multi-agent system that coordinates between a researcher and coder through a supervisor.
Based on https://colab.research.google.com/drive/1xDEPe2i_2rRqs7o6oNTtqA4J7Orsnvx1

Requires Tavily API Key https://github.com/tavily-ai/tavily-python
"""

import functools
import operator
from dataclasses import dataclass
from typing import Annotated, List, Sequence, TypedDict

from fi_instrumentation import register
from fi_instrumentation.fi_types import (
    EvalName,
    EvalSpanKind,
    EvalTag,
    EvalTagType,
    ProjectType,
)
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
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

# Configure trace provider with custom evaluation tags
trace_provider = register(
    project_type=ProjectType.EXPERIMENT,
    eval_tags=eval_tags,
    project_name="FUTURE_AGI",
    project_version_name="v1",
)

# Initialize the LangChain instrumentor
LangChainInstrumentor().instrument(tracer_provider=trace_provider)

tavily_tool = TavilySearchResults(max_results=5)

# This executes code locally, which can be unsafe
python_repl_tool = PythonREPLTool()


@dataclass
class AgentConfig:
    """Configuration for individual agents"""

    name: str
    system_prompt: str
    tools: List[any]


class AgentState(TypedDict):
    """The agent state that flows through the graph"""

    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


class AgentFactory:
    """Factory for creating and configuring agents"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def create_agent(self, config: AgentConfig) -> AgentExecutor:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", config.system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),

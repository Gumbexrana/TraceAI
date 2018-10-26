from typing import Annotated

from fi_instrumentation import register
from fi_instrumentation.fi_types import (
    EvalName,
    EvalSpanKind,
    EvalTag,
    EvalTagType,
    ProjectType,
)
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from traceai_langchain import LangChainInstrumentor
from typing_extensions import TypedDict

# Initialize database
db = SQLDatabase.from_uri("postgresql://user:password@localhost:5432/tfc")

# Create toolkit and get basic tools
toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(model="gpt-4o"))
tools = toolkit.get_tools()

list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")


# Define custom query tool
@tool
def db_query_tool(query: str) -> str:
    """Execute a SQL query against the database and get back the result."""
    result = db.run_no_throw(query)
    if not result:
        return "Error: Query failed. Please rewrite your query and try again."
    return result


# Define utility function for tool node fallbacks
def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks:
    """Create a ToolNode with error handling fallback."""
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def handle_tool_error(state):
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


# Define state and workflow
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


workflow = StateGraph(State)


# Define nodes
def first_tool_call(state: State):
    return {
        "messages": [
            AIMessage(
                content="Let me help you with that query. First, I'll check what tables are available.",
                tool_calls=[
                    {
                        "name": "sql_db_list_tables",
                        "args": {},
                        "id": "tool_list_tables",
                    }
                ],
            )
        ]
    }


# Add a handler for list tables response
def handle_list_tables(state: State):
    tables = next(
        msg.content
        for msg in reversed(state["messages"])
        if isinstance(msg, ToolMessage)
    )
    return {
        "messages": [
            AIMessage(
                content=f"I see the tables. Let me check their schema.",
                tool_calls=[
                    {
                        "name": "sql_db_schema",
                        "args": {},
                        "id": "tool_schema",

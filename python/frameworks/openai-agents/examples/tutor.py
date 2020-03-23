from asyncio import run

from agents import Agent, Runner
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from traceai_openai_agents import OpenAIAgentsInstrumentor

from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType

tracer_provider = register(project_name="openai-agents-example",
 project_type=ProjectType.OBSERVE)
OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)


history_tutor_agent = Agent(

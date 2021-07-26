import os

import vertexai
from fi_instrumentation import register
from fi_instrumentation.fi_types import (
    ProjectType,
)
from traceai_vertexai import VertexAIInstrumentor
from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
)

trace_provider = register(
    project_type=ProjectType.EXPERIMENT,
    project_name="FUTURE_AGI",
    project_version_name="v1",
)


VertexAIInstrumentor().instrument(tracer_provider=trace_provider)

# Initialize Vertex AI
vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location="us-central1")

# Initialize Gemini model
model = GenerativeModel("gemini-1.5-flash-002")

# Define the user's prompt in a Content object that we can reuse in model calls
user_prompt_content = Content(
    role="user",
    parts=[
        Part.from_text("What is the weather like in Boston?"),
    ],
)


import os

from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from mistralai import Mistral
from traceai_mistralai import MistralAIInstrumentor

trace_provider = register(
    project_type=ProjectType.EXPERIMENT,
    project_name="Future_AGI-MistralAI",
    project_version_name="v1",
)


MistralAIInstrumentor().instrument(tracer_provider=trace_provider)

# Load API key from environment or replace it directly
API_KEY = api_key = os.environ["MISTRAL_API_KEY"]

# Initialize Mistral Client
client = Mistral(api_key=API_KEY)
model = "mistral-large-latest"


# 1. **Chat Completion (Conversational AI)**
def chat_example():
    chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "What is the best French cheese?",
            },
        ],
    )
    return chat_response.choices[0].message.content


# 2. **Embeddings (Text Vector Representation)**
def embeddings_example():
    response = client.embeddings.create(
        model="mistral-embed",

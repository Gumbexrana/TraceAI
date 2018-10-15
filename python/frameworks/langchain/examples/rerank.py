
from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.schema import Document
from langchain.vectorstores import FAISS
from traceai_langchain import LangChainInstrumentor

trace_provider = register(
    project_type=ProjectType.EXPERIMENT,
    project_name="SQL_AGENT",
    project_version_name="V1",
)

LangChainInstrumentor().instrument(tracer_provider=trace_provider)

sample_docs = [
    Document(
        page_content="Artificial Intelligence (AI) is the simulation of human intelligence by machines. "
        "It includes machine learning, natural language processing, and robotics. "
        "AI systems can learn from experience, adjust to new inputs, and perform human-like tasks.",
        metadata={"source": "AI Introduction"},
    ),
    Document(
        page_content="Machine Learning is a subset of AI that enables systems to learn and improve "
        "from experience without being explicitly programmed. Popular ML algorithms include "
        "neural networks, decision trees, and random forests.",
        metadata={"source": "ML Basics"},
    ),
    Document(
        page_content="Python is a high-level programming language known for its simplicity and readability. "
        "It's widely used in data science, web development, and automation. Python has a rich ecosystem "
        "of libraries and frameworks.",
        metadata={"source": "Python Guide"},
    ),
    Document(
        page_content="Deep Learning is a type of machine learning based on artificial neural networks. "
        "It's particularly effective in tasks like image recognition, natural language processing, "

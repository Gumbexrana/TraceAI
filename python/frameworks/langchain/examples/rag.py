import os

from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereRerank
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from opentelemetry import trace
from opentelemetry.trace import set_tracer_provider
from traceai_langchain import LangChainInstrumentor

# Configure trace provider with custom evaluation tags
trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name="LANGCHAIN_RAG_OBSERVE",
    project_version_name="V1",
    # eval_tags=eval_tags,
)
set_tracer_provider(trace_provider)
# Initialize the LangChain instrumentor

# OpenAIInstrumentor().instrument(tracer_provider=trace_provider)
LangChainInstrumentor().instrument(tracer_provider=trace_provider)

tracer = trace.get_tracer(__name__)


def create_rag_pipeline(pdf_folder):
    # Load all PDF documents from the folder
    documents = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

    if not documents:
        raise ValueError(f"No PDF files found in {pdf_folder}")

    # Split text into chunks - using RecursiveCharacterTextSplitter for better handling of PDF content
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    texts = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(

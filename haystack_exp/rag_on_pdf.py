# %%
import os
from pathlib import Path
from dotenv import load_dotenv

from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.writers import DocumentWriter
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import ComponentDevice

# This env variable is needed for the SentenceTransformersDocumentEmbedder to stop throwing warning. Performance impact is unknown.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables (for OpenAI API key - still needed for LLM)
load_dotenv()

# Ensure the PDF file exists
pdf_path = Path("data/ram_awa_interview.pdf")
if not pdf_path.exists():
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

# Local embedding model configuration
MODEL_ID = "BAAI/bge-small-en-v1.5"  # or "thenlper/gte-small", "intfloat/e5-small-v2"
EMB_DIM = 384  # 384 for bge-small-en-v1.5, use 768 for larger models

# Initialize document store with correct parameters
document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

# Initialize indexing pipeline components
pdf_converter = PyPDFToDocument()
cleaner = DocumentCleaner(remove_empty_lines=True, remove_repeated_substrings=False)
splitter = DocumentSplitter(
    split_by="word",  # Split by word for better control
    split_length=200,  # Max number of words per chunk
    split_overlap=20,  # Number of words to overlap between chunks
)

# Replace OpenAI embedder with local SentenceTransformers embedder
embedder = SentenceTransformersDocumentEmbedder(
    model=MODEL_ID,
    device=ComponentDevice.from_str(
        "cpu"
    ),  # Use ComponentDevice.from_str("cuda") for GPU
    batch_size=32,
    normalize_embeddings=True,
)

writer = DocumentWriter(document_store)

# Build indexing pipeline
# The sequence of components is important:
# 1. PDF -> Documents (converter)
# 2. Clean documents (cleaner)
# 3. Split into chunks (splitter)
# 4. Generate embeddings (embedder)
# 5. Store in document store (writer)
indexing_pipeline = Pipeline()
indexing_pipeline.add_component("converter", pdf_converter)
indexing_pipeline.add_component("cleaner", cleaner)
indexing_pipeline.add_component("splitter", splitter)
indexing_pipeline.add_component("embedder", embedder)
indexing_pipeline.add_component("writer", writer)

# Connect the components in sequence
indexing_pipeline.connect("converter.documents", "cleaner.documents")
indexing_pipeline.connect("cleaner.documents", "splitter.documents")
indexing_pipeline.connect("splitter.documents", "embedder.documents")
indexing_pipeline.connect("embedder.documents", "writer.documents")

# Run the indexing pipeline
print("Starting PDF indexing with local embedding model...")
print(f"Using model: {MODEL_ID}")
indexing_result = indexing_pipeline.run(data={"sources": [str(pdf_path)]})
print(f"Indexed {indexing_result['writer']['documents_written']} document chunks")

# Initialize RAG pipeline components with local text embedder
text_embedder = SentenceTransformersTextEmbedder(
    model=MODEL_ID,
    device=ComponentDevice.from_str(
        "cpu"
    ),  # Use ComponentDevice.from_str("cuda") for GPU
    normalize_embeddings=True,
    prefix="",  # leave blank for BGE / GTE; use "query: " with e5
)

retriever = InMemoryEmbeddingRetriever(document_store)

# Define the prompt template for Q&A
prompt_template = [
    ChatMessage.from_user(
        """
        Given these documents from the RAM AWA interview, answer the question accurately and concisely.
        
        Documents:
        {% for doc in documents %}
            {{ doc.content }}
        {% endfor %}
        
        Question: {{query}}
        
        Answer based on the provided documents. If the information is not available in the documents, please say so.
        Answer:
        """
    )
]

prompt_builder = ChatPromptBuilder(template=prompt_template)
# default model is  model: str = "gpt-4o-mini",
# Still using OpenAI for text generation
llm = OpenAIChatGenerator()

# Build RAG pipeline
rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)

# Connect RAG pipeline components
rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")


# Function to ask questions
def ask_question(question: str) -> str:
    """
    Ask a question about the PDF content and get an answer.

    Args:
        question (str): The question to ask

    Returns:
        str: The answer from the RAG system
    """
    print(f"\nQuestion: {question}")
    result = rag_pipeline.run(
        data={
            "prompt_builder": {"query": question},
            "text_embedder": {"text": question},
        }
    )
    answer = result["llm"]["replies"][0].text
    print(f"Answer: {answer}")
    return answer


# Example questions about the RAM AWA interview
print("\n" + "=" * 50)
print("RAG System Ready - Ask questions about the RAM AWA interview")
print(f"Using local embedding model: {MODEL_ID}")
print("=" * 50)

# Sample questions - you can modify these or add your own
sample_questions = [
    "What is the main topic of this interview?",
    "Who is being interviewed?",
    "What are the key points discussed in the interview?",
    "What is RAM's role or position?",
    "What are the main challenges mentioned in the interview?",
]

# Ask sample questions
for question in sample_questions:
    ask_question(question)
    print("-" * 30)

# Interactive Q&A loop
print("\nEnter your questions (type 'quit' to exit):")
while True:
    user_question = input("\nYour question: ").strip()
    if user_question.lower() in ["quit", "exit", "q"]:
        break
    if user_question:
        ask_question(user_question)

print("RAG session ended.")

# %%

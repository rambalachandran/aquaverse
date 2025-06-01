# %%
import os
from pathlib import Path
from dotenv import load_dotenv

from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

# Load environment variables (for OpenAI API key)
load_dotenv()

# Ensure the PDF file exists
pdf_path = Path("data/ram_awa_interview.pdf")
if not pdf_path.exists():
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

# Initialize document store
document_store = InMemoryDocumentStore()

# Initialize indexing pipeline components
pdf_converter = PyPDFToDocument()
cleaner = DocumentCleaner(
    remove_empty_lines=True,
    remove_repeated_substrings=False
)
splitter = DocumentSplitter(
    split_by="word",      # Split by word for better control
    split_length=200,     # Max number of words per chunk
    split_overlap=20      # Number of words to overlap between chunks
)
embedder = OpenAIDocumentEmbedder()
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
print("Starting PDF indexing...")
indexing_result = indexing_pipeline.run(data={"sources": [str(pdf_path)]})
print(f"Indexed {indexing_result['writer']['documents_written']} document chunks")

# Initialize RAG pipeline components
text_embedder = OpenAITextEmbedder()
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
    result = rag_pipeline.run(data={
        "prompt_builder": {"query": question}, 
        "text_embedder": {"text": question}
    })
    answer = result["llm"]["replies"][0].text
    print(f"Answer: {answer}")
    return answer

# Example questions about the RAM AWA interview
print("\n" + "="*50)
print("RAG System Ready - Ask questions about the RAM AWA interview")
print("="*50)

# Sample questions - you can modify these or add your own
sample_questions = [
    "What is the main topic of this interview?",
    "Who is being interviewed?",
    "What are the key points discussed in the interview?",
    "What is RAM's role or position?",
    "What are the main challenges mentioned in the interview?"
]

# Ask sample questions
for question in sample_questions:
    ask_question(question)
    print("-" * 30)

# Interactive Q&A loop
print("\nEnter your questions (type 'quit' to exit):")
while True:
    user_question = input("\nYour question: ").strip()
    if user_question.lower() in ['quit', 'exit', 'q']:
        break
    if user_question:
        ask_question(user_question)

print("RAG session ended.")

# %%

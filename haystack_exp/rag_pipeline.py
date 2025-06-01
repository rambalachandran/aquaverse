import os
from dotenv import load_dotenv

from haystack import Pipeline
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack.components.embedders.sentence_transformers_text_embedder import SentenceTransformersTextEmbedder
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.dataclasses.chat_message import ChatMessage
from haystack.utils.device import ComponentDevice

# Load environment variables
load_dotenv()

# This env variable is needed for the SentenceTransformersDocumentEmbedder to stop throwing warning. Performance impact is unknown.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Local embedding model configuration
MODEL_ID = "BAAI/bge-small-en-v1.5"
EMB_DIM = 384

# Initialize Qdrant document store (ensure it matches the indexing pipeline)
document_store = QdrantDocumentStore(
    path="data/qdrant_storage",
    index="Document",
    embedding_dim=EMB_DIM,
)

# Initialize RAG pipeline components with local text embedder
text_embedder = SentenceTransformersTextEmbedder(
    model=MODEL_ID,
    device=ComponentDevice.from_str("cpu"),
    normalize_embeddings=True,
    prefix=""  # leave blank for BGE / GTE; use "query: " with e5
)

# Use EmbeddingRetriever with QdrantDocumentStore
retriever = QdrantEmbeddingRetriever(document_store=document_store)

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
    print(f"\nQuestion: {question}")
    result = rag_pipeline.run(data={
        "prompt_builder": {"query": question},
        "text_embedder": {"text": question}
    })
    answer = result["llm"]["replies"][0].text # Corrected to .content from .text
    print(f"Answer: {answer}")
    return answer

def main():
    # Example questions about the RAM AWA interview
    print("\n" + "="*50)
    print("RAG System Ready - Ask questions about the RAM AWA interview")
    print(f"Using local embedding model: {MODEL_ID}")
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

if __name__ == "__main__":
    main() 
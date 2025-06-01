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
from haystack.utils.auth import Secret
from typing import Optional # Keep for main function's api_key_for_cli

# Load environment variables
load_dotenv()

# This env variable is needed for the SentenceTransformersDocumentEmbedder to stop throwing warning. Performance impact is unknown.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Local embedding model configuration
MODEL_ID = "BAAI/bge-small-en-v1.5"
EMB_DIM = 384

# Initialize Qdrant document store
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

retriever = QdrantEmbeddingRetriever(document_store=document_store)

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

# Note: default_llm is removed as API key is now mandatory for ask_question.
# The main() function below will need to handle API key provision for its own testing.

# Function to ask questions - API key is now mandatory
def ask_question(question: str, api_key: str) -> str:
    pipeline_to_run = Pipeline()
    pipeline_to_run.add_component("text_embedder", text_embedder)
    pipeline_to_run.add_component("retriever", retriever)
    pipeline_to_run.add_component("prompt_builder", prompt_builder)

    # API key is now mandatory, so no 'else' branch needed here for default_llm
    user_llm = OpenAIChatGenerator(api_key=Secret.from_token(api_key))
    pipeline_to_run.add_component("llm", user_llm)

    pipeline_to_run.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline_to_run.connect("retriever.documents", "prompt_builder.documents")
    pipeline_to_run.connect("prompt_builder", "llm")
    
    result = pipeline_to_run.run(data={
        "prompt_builder": {"query": question},
        "text_embedder": {"text": question}
    })
    
    answer = result["llm"]["replies"][0].text 
    print(f"Answer: {answer}")
    return answer

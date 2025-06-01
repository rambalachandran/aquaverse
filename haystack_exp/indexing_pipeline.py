import os
from pathlib import Path
from dotenv import load_dotenv

from haystack import Pipeline
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.components.converters.pypdf import PyPDFToDocument
from haystack.components.preprocessors.document_cleaner import DocumentCleaner
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from haystack.components.embedders.sentence_transformers_document_embedder import SentenceTransformersDocumentEmbedder
from haystack.components.writers.document_writer import DocumentWriter
from haystack.utils.device import ComponentDevice

# This env variable is needed for the SentenceTransformersDocumentEmbedder to stop throwing warning. Performance impact is unknown.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

# Ensure the PDF file exists
pdf_path = Path("data/current_april_2025_vol_9.pdf")
if not pdf_path.exists():
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

# Local embedding model configuration
MODEL_ID = "BAAI/bge-small-en-v1.5"
EMB_DIM = 384

# Initialize Qdrant document store
document_store = QdrantDocumentStore(
    path="data/qdrant_storage",
    index="Document",
    embedding_dim=EMB_DIM,
    recreate_index=False,
    hnsw_config={"m": 16, "ef_construct": 64}
)

# Initialize indexing pipeline components
pdf_converter = PyPDFToDocument()
cleaner = DocumentCleaner(
    remove_empty_lines=True,
    remove_repeated_substrings=False
)
splitter = DocumentSplitter(
    split_by="word",
    split_length=200,
    split_overlap=20
)
embedder = SentenceTransformersDocumentEmbedder(
    model=MODEL_ID,
    device=ComponentDevice.from_str("cpu"),
    batch_size=32,
    normalize_embeddings=True
)
writer = DocumentWriter(document_store)

# Build indexing pipeline
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

def main():
    # Run the indexing pipeline
    print("Starting PDF indexing with local embedding model...")
    print(f"Using model: {MODEL_ID}")
    indexing_result = indexing_pipeline.run(data={"sources": [str(pdf_path)]})
    print(f"Indexed {indexing_result['writer']['documents_written']} document chunks")

if __name__ == "__main__":
    main() 
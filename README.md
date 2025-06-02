# AWA Current Magazine QnA

This project is a Question and Answering application built with Streamlit and Haystack. It allows users to ask questions about the content of the AWA Current Magazine and get answers based on a Retrieval Augmented Generation (RAG) system.

## Project Structure

- `app.py`: The main Streamlit application file. It handles the user interface, chat history, and interaction with the RAG pipeline.
- `index/`: Contains the indexing pipeline for processing and storing the magazine content.
  - `indexing_pipeline.py`: Defines and runs the Haystack pipeline to convert PDF documents, clean them, split them into manageable chunks, embed them using a sentence transformer model (`BAAI/bge-small-en-v1.5`), and write them to a Qdrant vector store.
- `rag/`: Contains the RAG pipeline for retrieving relevant documents and generating answers.
  - `rag_pipeline.py`: Defines the Haystack pipeline that takes a user's question, embeds it, retrieves relevant document chunks from the Qdrant store, builds a prompt with the retrieved context, and uses an OpenAI model (via `OpenAIChatGenerator`) to generate an answer.
- `data/`: This directory is expected to contain:
    - `current_april_2025_vol_9.pdf`: The source PDF document for the Q&A system. (Note: This file is not included in the repository and needs to be added by the user).
    - `qdrant_storage/`: The directory where the Qdrant vector store will save its data.

## Features

- **Streamlit Interface**: A simple and interactive web interface for asking questions and viewing conversation history.
- **RAG Pipeline**: Utilizes Haystack's RAG capabilities to provide context-aware answers.
- **Local Embedding Model**: Uses `BAAI/bge-small-en-v1.5` for document and query embeddings, running locally.
- **Qdrant Vector Store**: Stores and retrieves document embeddings efficiently.
- **OpenAI for Answer Generation**: Leverages OpenAI's chat models to generate answers based on the retrieved context.
- **Conversation History**: Keeps track of the last 10 questions and answers in the current session.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file is assumed. If it doesn't exist, you'll need to create one based on the imports in the Python files, e.g., `streamlit`, `haystack-ai`, `haystack-integrations`, `python-dotenv`, `sentence-transformers`, `qdrant-client`, `pypdf` etc.)*

4.  **Set up OpenAI API Key:**
    - The application requires an OpenAI API key. You can enter it directly in the sidebar of the Streamlit application.
    - Alternatively, you can create a `.env` file in the root directory and add your key:
      ```env
      OPENAI_API_KEY="your_openai_api_key_here"
      ```
      The `rag_pipeline.py` and `indexing_pipeline.py` use `python-dotenv` to load environment variables, though the primary way for `app.py` to get the key is via user input.

5.  **Prepare Data:**
    - Create a directory named `data` in the root of the project.
    - Place the `current_april_2025_vol_9.pdf` file (or your target PDF) into the `data/` directory.

## Running the Application

1.  **Run the Indexing Pipeline (First time setup or when the PDF changes):**
    Navigate to the `index` directory and run the indexing script. This will process your PDF and populate the Qdrant vector store.
    ```bash
    python index/indexing_pipeline.py
    ```
    This script will create a `qdrant_storage` subdirectory in the `data/` directory.

2.  **Run the Streamlit Application:**
    Once the indexing is complete, you can run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
    This will open the application in your web browser.

## How it Works

### Indexing (`index/indexing_pipeline.py`)

1.  **Load PDF**: The `PyPDFToDocument` component loads the PDF document.
2.  **Clean Document**: `DocumentCleaner` removes empty lines.
3.  **Split Document**: `DocumentSplitter` breaks the document into smaller, overlapping chunks (200 words with 20 words overlap).
4.  **Embed Documents**: `SentenceTransformersDocumentEmbedder` uses the `BAAI/bge-small-en-v1.5` model to create vector embeddings for each document chunk.
5.  **Write to Store**: `DocumentWriter` saves these embeddings and their corresponding text into the `QdrantDocumentStore`.

### Retrieval and Generation (`rag/rag_pipeline.py` and `app.py`)

1.  **User Input**: The user enters a question in the Streamlit UI.
2.  **Embed Query**: The `SentenceTransformersTextEmbedder` (within `ask_question` function) converts the user's question into a vector embedding using the same `BAAI/bge-small-en-v1.5` model.
3.  **Retrieve Documents**: `QdrantEmbeddingRetriever` searches the Qdrant store for document chunks whose embeddings are most similar to the query embedding.
4.  **Build Prompt**: `ChatPromptBuilder` constructs a prompt for the language model, incorporating the retrieved document chunks as context along with the user's original question.
5.  **Generate Answer**: `OpenAIChatGenerator` sends the prompt to an OpenAI chat model (e.g., GPT-3.5 Turbo, GPT-4) which generates an answer.
6.  **Display Answer**: The Streamlit app displays the generated answer and updates the conversation history.

## Configuration

-   **`MAX_HISTORY_LENGTH`** (`app.py`): Controls the number of recent Q&A pairs stored in the session history (default: 10).
-   **Embedding Model (`MODEL_ID`)**: Both indexing and RAG pipelines use `BAAI/bge-small-en-v1.5`. This can be changed in `index/indexing_pipeline.py` and `rag/rag_pipeline.py`. Remember to update `EMB_DIM` (embedding dimension) if you change the model.
-   **Qdrant Configuration**: Path and other Qdrant settings can be adjusted in `index/indexing_pipeline.py` and `rag/rag_pipeline.py`.
-   **PDF Path**: The path to the source PDF is hardcoded in `index/indexing_pipeline.py`.
-   **Tokenizer Parallelism**: The environment variable `TOKENIZERS_PARALLELISM` is set to `"false"` in both pipeline scripts to avoid warnings from the `sentence-transformers` library.

## To-Do / Potential Improvements

-   [ ] Create a `requirements.txt` file.
-   [ ] Make the PDF file path configurable (e.g., through UI or environment variable).
-   [ ] Allow users to upload PDFs directly through the UI.
-   [ ] Implement more sophisticated document preprocessing and chunking strategies.
-   [ ] Add error handling and logging.
-   [ ] Option to choose different LLMs or local LLMs for generation.
-   [ ] More detailed UI for displaying retrieved document sources.
-   [ ] Unit and integration tests.

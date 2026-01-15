# 📚 RAG SRS Generator

An intelligent system for generating Software Requirements Specification (SRS) documents using Retrieval-Augmented Generation (RAG). It leverages a Vector DB (ChromaDB) to retrieve relevant domain knowledge and an AI Judge to evaluate the quality of the generated SRS.

## 🚀 Features

*   **RAG-Powered Generation**: Checks knowledge base before writing to ensure accuracy.
*   **Hybrid Search**: Combines Semantic Search (Embeddings) + Keyword Search (BM25) + Reranking (CrossEncoder).
*   **AI Evaluation**: Automatically scores SRS on Completeness, Consistency, Accuracy, and Faithfulness.
*   **Interactive UI**: Streamlit dashboard for generation, debugging, and benchmarks.
*   **Benchmarks**: Built-in scripts for A/B testing (RAG vs Pure LLM) and Retrieval Metrics (Hit Rate, MRR).

## 🛠️ Tech Stack

*   **Backend**: FastAPI, Python 3.12, UV (Package Manager).
*   **AI/ML**: Google Gemini (Generation), OpenAI/HuggingFace (Embeddings), ChromaDB (Vector Store).
*   **Frontend**: Streamlit, Altair (Charts), Mermaid (Diagrams).

## ⚙️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/TNTD-dev/test-rag-model-srs.git
    cd test-rag-model-srs
    ```

2.  **Install dependencies**:
    ```bash
    uv sync
    ```

3.  **Environment Setup**:
    Copy `env.example` to `.env` and configure your API keys:
    ```bash
    cp env.example .env
    # Edit .env with your GEMINI_API_KEY, OPENAI_API_KEY (if used), etc.
    ```

## 🏃 Usage

### 1. Start the System
Open 2 terminals:

**Terminal 1: Backend API**
```bash
uv run uvicorn src.app.main:app --reload
```
*Docs available at: http://localhost:8000/docs*

**Terminal 2: Frontend UI**
```bash
uv run streamlit run streamlit_app.py
```
*Access UI at: http://localhost:8501*

### 2. Run Benchmarks

**Retrieval Benchmark (Hit Rate/MRR)**:
```bash
uv run python generate_testset.py      # Generate synthetic test data
uv run python benchmark_retrieval.py   # Run evaluation
```

**A/B Testing (RAG vs Vanilla)**:
```bash
uv run python benchmark_ab.py
```

## 📂 Project Structure
*   `src/app/services`: Core logic (RAG Retriever, Indexer, Generator, Evaluator).
*   `data/`: Knowledge base documents (Markdown).
*   `rag_db/`: Local ChromaDB vector store.
*   `benchmark_*.py`: Scripts for performance testing.

---
*Created by TNTD-dev*

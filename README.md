# 💬 EBook AI Navigator

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FC60A8?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.trychroma.com)
[![Model Context Protocol](https://img.shields.io/badge/MCP-Web_Search-007ACC?style=for-the-badge&logo=protocols)](https://modelcontextprotocol.io)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**EBook AI Navigator** is an advanced, decoupled Retrieval-Augmented Generation (RAG) system built to query ebooks, manuals, and documents (PDF/TXT) through a polished, responsive ChatGPT-like interface. 

The application architecture features an **intelligent hybrid orchestrator** that dynamically classifies user intent, routing queries between **local vector embeddings (ChromaDB)** and **real-time web search (DuckDuckGo)** using the revolutionary **Model Context Protocol (MCP)**. Powered by high-performance **FastAPI** on the backend and an interactive **Streamlit** client, it offers conversational memory, customizable system prompts, and full transparency into the model's step-by-step thinking process.

---

## 🌟 Key Features

*   **🔌 Decoupled Web Architecture:** Fully isolated backend REST API (FastAPI) and frontend UI (Streamlit) communicating via clean, structured JSON endpoints.
*   **🧠 Intelligent Query Orchestrator:** Automatically classifies queries into three paths:
    *   `rag`: Answers solely from local vector database context.
    *   `web`: Answers from real-time web search (perfect for current trends or fresh data).
    *   `both`: Synthesizes a unified response combining local docs and live web search.
*   **🔌 Model Context Protocol (MCP) Integration:** Features an integrated MCP client that dynamically spawns a `duckduckgo-mcp-server` over `stdio` transport. This allows secure, sandboxed, protocol-based web searches to enrich context.
*   **📉 Smart Confidence-Based Fallback:** If RAG confidence (vector similarity score) falls below the configurable threshold, the system automatically redirects the query to a web search to prevent hallucination.
*   **📚 Dynamic Document Ingestion:** Supports index ingestion for plain text (`.txt`) and PDF (`.pdf`) documents. Features an online file-uploader that updates the active collection in real-time.
*   **🧠 Semantic Chunking & Local Embeddings:** Employs `RecursiveCharacterTextSplitter` (chunk size 800, overlap 150) to preserve sentences and uses `SentenceTransformers` (`all-MiniLM-L6-v2`) for local hardware-accelerated vector indexing.
*   **💬 Rolling Conversation Memory & Deep Reasoning:** Maintains a full conversational dialog stack. Displays backend model reasoning processes step-by-step inside collapsible UI panels (`<think>...</think>` parse rendering).

---

## 🏗️ Architecture Overview

The EBook AI Navigator clearly separates the ingestion path from the intelligent query routing and synthesis pipeline:

```mermaid
graph TD
    classDef frontend fill:#E8F0FE,stroke:#1A73E8,stroke-width:1px,color:#000000;
    classDef api fill:#E6F4EA,stroke:#137333,stroke-width:1px,color:#000000;
    classDef processing fill:#FEF7E0,stroke:#B06000,stroke-width:1px,color:#000000;
    classDef database fill:#FCE8E6,stroke:#C5221F,stroke-width:1px,color:#000000;
    classDef external fill:#F3E8FD,stroke:#8430CE,stroke-width:1px,color:#000000;
    classDef mcp fill:#E3F2FD,stroke:#0D47A1,stroke-width:1px,color:#000000;

    ST[Streamlit Frontend]:::frontend ==>|HTTP REST Request| F_API[FastAPI Backend Server]:::api
    
    subgraph Ingestion Path (Write)
        F_API ==>|Upload File| INGEST[ingest.py]:::processing
        INGEST ==>|Text Extraction & Split| EMBED[SentenceTransformer]:::processing
        EMBED ==>|Vector Arrays| CHROMA[(ChromaDB Collection)]:::database
    end

    subgraph Intelligent Routing & Query Path (Read)
        F_API ==>|Ask Question| RAG[rag_pipeline.py]:::processing
        RAG ==>|Check Config| ORCH[orchestrator.py]:::processing
        ORCH ==>|1. Classify Route: RAG/WEB/BOTH| LLM_ROUTE[Sarvam AI API]:::external
        ORCH ==>|2. If RAG/BOTH| CHROMA
        ORCH ==>|3. If WEB/BOTH| MCP[mcp_client.py]:::mcp
        MCP <==>|Stdio Protocol| DDG_SERVER[DuckDuckGo MCP Server]:::mcp
        ORCH ==>|4. Synthesis Prompt| LLM_SYNTH[Sarvam AI API]:::external
    end
```

---

## ⚙️ Configuration Setup

Create a `.env` file at the root of the project to manage environment keys and fine-tune your pipeline configurations:

```env
# Sarvam AI Credentials
SARVAM_API_KEY=your_sarvam_api_key_here

# Intelligent Routing Settings
ORCHESTRATOR_ENABLED=true
RAG_CONFIDENCE_THRESHOLD=0.5
DDGO_MAX_RESULTS=5

# Network Settings (Optional)
BACKEND_URL=http://127.0.0.1:8000
```

### ⚙️ Available Settings Details
*   `SARVAM_API_KEY`: API credential key from [Sarvam AI](https://www.sarvam.ai/).
*   `ORCHESTRATOR_ENABLED` (`true`/`false`): Turn intent-based classification and MCP web-search fallback on or off. If set to `false`, the system acts as a standard offline RAG pipeline.
*   `RAG_CONFIDENCE_THRESHOLD` (`0.0` - `1.0`): The minimal document vector similarity score required to trust local answers. If the best match score falls below this, the system queries the DuckDuckGo MCP server.
*   `DDGO_MAX_RESULTS`: The maximum number of search result summaries fetched from the web search protocol.

---

## 🚀 Installation & Running

### 1. Prerequisites
Ensure you have **Python 3.10** or higher installed.

### 2. Setup Virtual Environment & Dependencies
```bash
# Clone the repository
git clone https://github.com/your-username/ebook-ai-navigator.git
cd ebook-ai-navigator

# Create and activate a virtual environment
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Running the App via the Command Line Interface (CLI)
The project includes a unified orchestrator CLI (`main.py`) to manage ingestion, command-line testing, and combined servers:

```bash
# 1. Ingest all local files (.pdf, .txt) in the data/ directory into ChromaDB
python main.py --ingest

# 2. Run a one-off quick query directly in your shell terminal
python main.py --query "What is Retrieval Augmented Generation?"

# 3. Launch an interactive, memory-retaining chat session inside your terminal
python main.py --chat

# 4. Spin up only the background FastAPI backend server
python main.py --api
```

### 4. Running the Decoupled Web Interface
To launch **both** the FastAPI REST backend server and the Streamlit frontend client together in a unified manager:

```bash
python main.py --serve
```

Once launched:
*   **Streamlit Web Client:** Automatically opens in your browser at `http://localhost:8501`.
*   **FastAPI Backend Server:** Running at `http://localhost:8000` (interactive API documentation is accessible at `http://localhost:8000/docs`).

---

## 🛠️ API Documentation

The FastAPI backend exposes the following RESTful endpoints:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/config` | `GET` | Returns active system settings (embedding models, LLM engine, and routing status). |
| `/files` | `GET` | Lists all active documents currently stored in the `data/` directory. |
| `/upload` | `POST` | Uploads a new PDF/TXT file and triggers the automated ingestion and vector indexing pipeline. |
| `/query` | `POST` | Processes a prompt incorporating chat history, routing intent, and custom system instructions. |

---

## 📂 Project Structure

```
ebook-rag/
├── app/
│   └── streamlit_app.py     # Streamlit front-end client interface
├── data/
│   └── ebook.txt            # Data drop directory for TXT/PDF documents
├── chroma_db/               # Local SQLite-backed Chroma vector store
├── src/
│   ├── __init__.py
│   ├── api.py               # FastAPI REST endpoints
│   ├── config.py            # Global environment variables and thresholds
│   ├── ingest.py            # Semantic chunking, PDF loading, and indexing
│   ├── mcp_client.py        # Model Context Protocol stdio client
│   ├── orchestrator.py      # LLM classifier routing and search orchestration
│   ├── prompts.py           # Core system, router, and aggregator prompts
│   ├── rag_pipeline.py      # Base retrieval and pipeline entry point
│   └── vectordb.py          # Sentence-transformers and persistent ChromaDB helpers
├── main.py                  # Unified project orchestration CLI
├── requirements.txt         # Package dependencies
└── README.md                # System documentation
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

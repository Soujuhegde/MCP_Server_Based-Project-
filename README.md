# 💬 EBook AI Navigator

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FC60A8?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.trychroma.com)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**EBook AI Navigator** is a decoupled Retrieval-Augmented Generation (RAG) system built to query ebooks and documents in standard PDF or Text formats through a polished ChatGPT-like interface. 

The application architecture decouples the high-performance **FastAPI Rest API** backend from the minimalist **Streamlit** user interface. Behind the scenes, documents undergo semantic text chunking, local vector embeddings, and persistent spatial queries via **ChromaDB**, yielding contexts synthesized into concise dialog answers by **Sarvam AI**.

---

## 🌟 Key Features

*   **🔌 Decoupled Web Architecture:** Fully isolated frontend (Streamlit) and backend REST API (FastAPI) which communicate entirely via structured JSON endpoints.
*   **📚 Dynamic Document Ingestion:** Supports indexing plain text (`.txt`) and PDF (`.pdf`) documents directly via physical filesystem drops or online drag-and-drop uploads.
*   **🧠 Semantic Text Chunking:** Employs a `RecursiveCharacterTextSplitter` algorithm to preserve structural sentences and overlap contexts, mitigating context loss.
*   **⚡ Local Vector Search:** Leverages `SentenceTransformers` (`all-MiniLM-L6-v2`) to perform hardware-accelerated local vector embedding indexing inside a persistent **ChromaDB** store.
*   **💬 Conversation Memory & Deep Reasoning:** Maintains a rolling system dialog stack, supplying temporal history alongside retrieved context to support conversational follow-ups. Displays backend model thinking process step-by-step.

---

## 🏗️ Architecture Overview

The system divides the read and write paths clearly:

```mermaid
graph TD
    classDef frontend fill:#E8F0FE,stroke:#1A73E8,stroke-width:1px,color:#000000;
    classDef api fill:#E6F4EA,stroke:#137333,stroke-width:1px,color:#000000;
    classDef processing fill:#FEF7E0,stroke:#B06000,stroke-width:1px,color:#000000;
    classDef database fill:#FCE8E6,stroke:#C5221F,stroke-width:1px,color:#000000;
    classDef external fill:#F3E8FD,stroke:#8430CE,stroke-width:1px,color:#000000;

    ST[Streamlit Frontend]:::frontend ==>|HTTP REST Request| F_API[FastAPI Backend Server]:::api
    
    subgraph Ingestion Path (Write)
        F_API ==>|Upload File| INGEST[ingest.py]:::processing
        INGEST ==>|Text Extraction & Split| EMBED[SentenceTransformer]:::processing
        EMBED ==>|Vector Arrays| CHROMA[(ChromaDB Collection)]:::database
    end

    subgraph Query Path (Read)
        F_API ==>|Ask Question| RAG[rag_pipeline.py]:::processing
        RAG ==>|Query Similarity Search| CHROMA
        RAG ==>|Formulate Context Prompt| LLM[Sarvam AI API]:::external
        LLM -->> RAG
    end
```

---

## ⚙️ Configuration Setup

Create a `.env` file at the root of the project to declare your environment keys:

```env
# EBook AI Navigator configuration
SARVAM_API_KEY=your_sarvam_api_key_here
```

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
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Running the App via the Command Line Interface (CLI)

The repository provides a consolidated orchestrator (`main.py`) which manages ingestion, command-line query systems, backend hosting, and combined launchers.

```bash
# Ingest local files inside data/ directory into ChromaDB
python main.py --ingest

# Run a quick query directly from your command-line console
python main.py --query "What are the core concepts covered in the book?"

# Launch an interactive chat session inside your terminal
python main.py --chat
```

### 4. Running the Decoupled Web Interface

To launch both the FastAPI REST backend server and the Streamlit frontend client seamlessly:

```bash
python main.py --serve
```

Once running:
*   **FastAPI Backend URL:** `http://localhost:8000` (API documentation accessible via `http://localhost:8000/docs`)
*   **Streamlit Web Client:** Accessible via the local address outputted by Streamlit (typically `http://localhost:8501`)

---

## 🛠️ API Documentation

The FastAPI backend exposes the following clean endpoints:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/config` | `GET` | Returns system config (embedding model, LLM engine). |
| `/files` | `GET` | Lists all active files (.pdf, .txt) inside the `data/` directory. |
| `/upload` | `POST` | Uploads a new document and runs the ingestion pipeline automatically. |
| `/query` | `POST` | Queries the RAG pipeline incorporating conversational context history. |

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

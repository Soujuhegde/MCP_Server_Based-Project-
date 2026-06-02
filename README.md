# 💬 EBook AI Navigator

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FC60A8?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.trychroma.com)
[![MCP](https://img.shields.io/badge/MCP-Web_Search-007ACC?style=for-the-badge&logo=protocols&logoColor=white)](https://modelcontextprotocol.io)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Sarvam AI](https://img.shields.io/badge/LLM-Sarvam_AI-8B5CF6?style=for-the-badge&logo=openai&logoColor=white)](https://www.sarvam.ai/)

**EBook AI Navigator** is a state-of-the-art, fully decoupled Retrieval-Augmented Generation (RAG) system engineered to turn static ebooks, manuals, and PDF/TXT documents into dynamic, conversational intelligence. 

Featuring a modern **intelligent query orchestrator**, the system dynamically classifies user intent, routing requests between **local vector embeddings (ChromaDB)** and **real-time web search (DuckDuckGo)** using the cutting-edge **Model Context Protocol (MCP)**. Powered by high-performance **FastAPI** on the backend and a premium, responsive **Streamlit** user interface, it provides complete conversational memory, custom system guidelines, and full transparency into the model's step-by-step thinking process.

---

## 🌟 Key Features

*   **🔌 Decoupled Microservices Architecture:** Fully separated backend API (FastAPI) and frontend (Streamlit) communicating over clean RESTful endpoints.
*   **🧠 Intelligent Query Orchestrator:** Powered by `sarvam-m`, the system dynamically routes incoming prompts based on semantic intent:
    *   `rag`: Restricts context retrieval to your uploaded documents only.
    *   `web`: Queries the live internet for real-time information or fresh updates.
    *   `both`: Synthesizes a unified response combining local documents with real-time web results.
*   **🔌 Integrated Model Context Protocol (MCP):** Spawns an internal, sandboxed `duckduckgo-mcp-server` over `stdio` transport. The client dynamically establishes high-speed, protocol-compliant web searches without external API keys.
*   **📉 Smart Confidence-Based Fallback:** If RAG search confidence (vector similarity score) drops below the configurable threshold, the engine automatically escalates the query to a web search, ensuring hallucination-free answers.
*   **📚 Dynamic Document Ingestion:** Supports indexing text (`.txt`) and PDF (`.pdf`) documents. Features an interactive, drag-and-drop file uploader that updates vector store collections in real-time.
*   **🧬 Semantic Chunking & Local Embeddings:** Employs recursive semantic chunking (`RecursiveCharacterTextSplitter` with 800-character size and 150-character overlap) for context integrity, powered by a hardware-accelerated **SentenceTransformers** (`all-MiniLM-L6-v2`) model run locally.
*   **💬 Rolling Memory & Deep Reasoning:** Incorporates full conversation state management. It extracts and displays the backend model's step-by-step reasoning processes inside beautiful, collapsible UI panels (`<think>...</think>` parsing).

---

## 🏗️ Architecture Overview

The system divides the data workflow into two distinct paths: a high-efficiency **Ingestion Path** and a dynamic, multi-agent **Query Execution & Synthesis Path**:

```mermaid
graph TD
    %% Define CSS classes with elegant colors
    classDef frontend fill:#1E293B,stroke:#3B82F6,stroke-width:2px,color:#F8FAFC;
    classDef api fill:#0F172A,stroke:#10B981,stroke-width:2px,color:#F8FAFC;
    classDef processing fill:#1E1B4B,stroke:#6366F1,stroke-width:2px,color:#F8FAFC;
    classDef database fill:#1C1917,stroke:#F59E0B,stroke-width:2px,color:#F8FAFC;
    classDef external fill:#1F1625,stroke:#D946EF,stroke-width:2px,color:#F8FAFC;
    classDef mcp fill:#172554,stroke:#0EA5E9,stroke-width:2px,color:#F8FAFC;

    %% Ingestion Flow
    subgraph Ingestion Pipeline ["📥 Semantic Document Ingestion"]
        direction LR
        DOCS[Document Uploads <br> .pdf, .txt]:::processing -->|Loader| PARSE[PyPDF / UTF-8 Reader]:::processing
        PARSE -->|Text Splitter| SPLIT[Recursive Splitter <br> 800 char chunk / 150 overlap]:::processing
        SPLIT -->|Local Embeddings| EMBED[SentenceTransformer <br> all-MiniLM-L6-v2]:::processing
        EMBED -->|Vector Matrix| CHROMA[(Chroma Vector DB <br> Persistent Collection)]:::database
    end

    %% Execution Query Flow
    subgraph Query Execution ["🧠 Hybrid Query Execution & Synthesis"]
        direction TB
        UI[Streamlit Web Client]:::frontend <==>|JSON REST API| BACKEND[FastAPI Backend Server]:::api
        BACKEND <==>|Invoke Pipeline| PIPELINE[rag_pipeline.py]:::processing
        
        PIPELINE ==>|1. Query Similarity Search| CHROMA
        CHROMA ==>|2. Document Chunks & Scores| PIPELINE
        
        PIPELINE ==>|3. Parse Intent| ORCH[orchestrator.py]:::processing
        ORCH ==>|4. Classify Route| ROUTER_LLM[Sarvam AI LLM <br> model: sarvam-m]:::external
        
        ROUTER_LLM ==>|Intent: rag / web / both| ORCH
        
        %% Conditional Routes
        ORCH -.->|RAG Confidence < Threshold <br> OR Intent: web/both| MCP_CLIENT[mcp_client.py]:::mcp
        MCP_CLIENT <==>|Stdio Protocol| DDG_MCP[duckduckgo-mcp-server]:::mcp
        
        ORCH ==>|5. Combined Prompt Context| SYNTH_LLM[Sarvam AI LLM <br> Synthesized Generation]:::external
        SYNTH_LLM ==>|6. Response with reasoning| BACKEND
    end

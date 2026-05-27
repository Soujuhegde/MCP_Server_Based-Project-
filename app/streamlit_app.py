import streamlit as st
import sys
from pathlib import Path
import os
import requests

# Backend API server URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


def parse_reasoning_and_answer(text):
    """Parses thinking block <think>...</think> from the LLM output if present."""
    import re
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()
        # Remove the think block from the final answer
        answer = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        return reasoning, answer
    return None, text


# Set up page configurations
st.set_page_config(
    page_title="EBook AI Navigator",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Font Overrides */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
}

/* Gradient Title */
.gradient-text {
    background: linear-gradient(135deg, #FF4B4B, #FF8F8F);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.6rem;
    margin-bottom: 0px;
    padding-bottom: 0px;
}

.subtitle-text {
    font-size: 0.95rem;
    color: var(--text-color);
    opacity: 0.75;
    margin-top: -4px;
    margin-bottom: 25px;
}

/* Sidebar Customizations */
[data-testid="stSidebar"] {
    background-color: var(--secondary-background-color);
    border-right: 1px solid rgba(128, 128, 128, 0.12);
}

[data-testid="stSidebar"] h3 {
    font-family: 'Outfit', sans-serif;
    color: var(--text-color);
    font-size: 1.05rem;
    font-weight: 700;
    margin-top: 18px;
    margin-bottom: 10px;
    padding-bottom: 4px;
    border-bottom: 1px solid rgba(128, 128, 128, 0.15);
}

/* Custom card container for Sidebar items */
.config-container {
    background-color: var(--background-color);
    border: 1px solid rgba(128, 128, 128, 0.15);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 10px;
    font-size: 0.8rem;
    line-height: 1.5;
}

/* Active documents list item style */
.active-file-item {
    font-size: 0.78rem;
    font-family: monospace;
    background-color: var(--background-color);
    border: 1px solid rgba(128, 128, 128, 0.1);
    border-radius: 6px;
    padding: 6px 10px;
    margin-bottom: 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Custom Route Badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 12px;
}

.badge-rag {
    background-color: rgba(26, 115, 232, 0.12);
    color: #1a73e8;
    border: 1px solid rgba(26, 115, 232, 0.25);
}

.badge-web {
    background-color: rgba(249, 171, 0, 0.12);
    color: #e37400;
    border: 1px solid rgba(249, 171, 0, 0.25);
}

.badge-both {
    background-color: rgba(168, 85, 247, 0.12);
    color: #a855f7;
    border: 1px solid rgba(168, 85, 247, 0.25);
}

/* Chunk and Web Search Citations Card */
.citation-container {
    background-color: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.12);
    border-radius: 10px;
    padding: 12px 14px;
    margin-top: 8px;
    margin-bottom: 4px;
    transition: all 0.2s ease;
}

.citation-container:hover {
    border-color: rgba(128, 128, 128, 0.25);
}

.citation-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.78rem;
    font-weight: 600;
    opacity: 0.8;
    margin-bottom: 6px;
    border-bottom: 1px solid rgba(128, 128, 128, 0.1);
    padding-bottom: 4px;
}

.citation-body {
    font-size: 0.86rem;
    line-height: 1.5;
    font-style: italic;
    color: var(--text-color);
}

/* Custom thinking block style */
.thinking-container {
    background-color: rgba(128, 128, 128, 0.03);
    border-left: 3px solid var(--primary-color);
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin-bottom: 16px;
}

.thinking-title {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
    color: var(--primary-color);
}

.thinking-content {
    font-size: 0.88rem;
    line-height: 1.5;
    font-family: monospace;
    opacity: 0.85;
    white-space: pre-wrap;
}

/* Micro-interactions on buttons */
.stButton>button {
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
    font-weight: 500 !important;
}

.stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)

# Header Area
st.markdown('<div class="gradient-text">💬 EBook AI Navigator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">An elegant hybrid RAG search agent powered by Model Context Protocol (MCP)</div>', unsafe_allow_html=True)

# Initialize Session State Variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()

# Fetch settings config from backend
try:
    response = requests.get(f"{BACKEND_URL}/config", timeout=5)
    if response.status_code == 200:
        backend_config = response.json()
    else:
        backend_config = {}
except Exception:
    backend_config = {}

embedding_model = backend_config.get("embedding_model", "Unknown (Backend offline)")
llm_engine = backend_config.get("llm_engine", "Unknown (Backend offline)")
database = backend_config.get("database", "ChromaDB (Persistent)")
orchestrator_enabled = backend_config.get("orchestrator_enabled", True)

# Setup Sidebar Control Room
with st.sidebar:
    st.markdown("### ⚙️ Engine Control Room")

    # Dynamic settings info
    status_color = "#10b981" if orchestrator_enabled and "Unknown" not in llm_engine else "#ef4444"
    status_text = "Enabled" if orchestrator_enabled and "Unknown" not in llm_engine else "Disabled/Offline"

    st.markdown(f"""
    <div class="config-container">
        <b>LLM Core:</b> <code>{llm_engine}</code><br/>
        <b>Vector DB:</b> <code>{database}</code><br/>
        <b>Embeddings:</b> <code>{embedding_model}</code><br/>
        <b>Router:</b> <code style="color: {status_color}; font-weight: bold;">{status_text}</code>
    </div>
    """, unsafe_allow_html=True)

    # Document Indexer
    st.markdown("### 📥 Index Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT to ingest:",
        type=["pdf", "txt"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if file_key not in st.session_state.ingested_files:
            with st.spinner("💾 Ingesting & indexing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    r = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=120)

                    if r.status_code == 200:
                        st.session_state.ingested_files.add(file_key)
                        st.success(f"Indexed '{uploaded_file.name}'!")
                        st.session_state.messages = []  # Reset chat history on new upload
                        st.rerun()
                    else:
                        try:
                            err_detail = r.json().get('detail', 'Unknown API Error')
                        except Exception:
                            err_detail = r.text
                        st.error(f"Failed to ingest: {err_detail}")
                except Exception as e:
                    st.error(f"Backend offline: {str(e)}")

    # Active Files Panel
    st.markdown("### 📄 Active Collections")
    try:
        r = requests.get(f"{BACKEND_URL}/files", timeout=5)
        if r.status_code == 200:
            files = r.json().get("files", [])
            if files:
                for f in files:
                    st.markdown(f'<div class="active-file-item">📄 {f}</div>', unsafe_allow_html=True)
            else:
                st.caption("No files in the knowledge base.")
        else:
            st.caption("Unable to load index collections.")
    except Exception:
        st.caption("Backend offline.")

    # Custom Instructions Panel
    st.markdown("### 📝 System Instructions")
    user_instructions = st.text_area(
        "System instructions or guidelines:",
        placeholder="e.g., Explain like I'm 5, Keep it concise...",
        height=90,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Clear button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.ingested_files = set()
        st.success("Conversation cleared!")
        st.rerun()

# Display rolling chat messages from session state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            # 1. Display Routing Badge
            route = msg.get("route_used")
            if route:
                if route == "rag":
                    st.markdown('<div class="badge badge-rag">🎯 Route: Local Vector DB</div>', unsafe_allow_html=True)
                elif route == "web":
                    st.markdown('<div class="badge badge-web">🌐 Route: Web Search (MCP)</div>', unsafe_allow_html=True)
                elif route == "both":
                    st.markdown('<div class="badge badge-both">⚡ Route: Hybrid (DB + Web)</div>', unsafe_allow_html=True)

            # 2. Extract and display Reasoning block
            reasoning, clean_content = parse_reasoning_and_answer(msg["content"])
            if reasoning:
                with st.expander("🧠 View Thinking Process"):
                    st.markdown(
                        f'<div class="thinking-container"><div class="thinking-title">Thinking Process</div><div class="thinking-content">{reasoning}</div></div>',
                        unsafe_allow_html=True
                    )

            # 3. Display main answer content
            st.write(clean_content)

            # 4. Display local retrieved chunks if present
            if "documents" in msg and msg["documents"]:
                with st.expander("🎯 View Retrieved Document Chunks"):
                    for idx, doc in enumerate(msg["documents"]):
                        source = doc.get("metadata", {}).get("source", "Unknown Source") if doc.get("metadata") else "Unknown Source"
                        similarity = doc.get("similarity_score", 0.0)

                        st.markdown(f"""
                        <div class="citation-container">
                            <div class="citation-header">
                                <span>📄 Source {idx+1}: {source}</span>
                                <span style="color: #1a73e8; font-weight:700;">Match: {similarity*100:.1f}%</span>
                            </div>
                            <div class="citation-body">"{doc['text']}"</div>
                        </div>
                        """, unsafe_allow_html=True)

            # 5. Display web search results if present
            if "web_results" in msg and msg["web_results"]:
                with st.expander("🌐 View Web Search References (MCP)"):
                    st.write(msg["web_results"])
        else:
            st.write(msg["content"])

# Prompt input at the bottom of the page
if prompt := st.chat_input("Ask a question about your documents..."):
    # Render user prompt
    with st.chat_message("user"):
        st.write(prompt)

    # Append user prompt to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                # Pass past messages as context via API payload
                payload = {
                    "query": prompt,
                    "chat_history": st.session_state.messages[:-1],
                    "user_instructions": user_instructions if user_instructions.strip() else None
                }
                r = requests.post(f"{BACKEND_URL}/query", json=payload, timeout=120)

                if r.status_code == 200:
                    result = r.json()
                    response = result["answer"]
                    docs = result.get("documents", [])
                    route_used = result.get("route_used")
                    web_results = result.get("web_results")

                    # Render response
                    # 1. Badge
                    if route_used:
                        if route_used == "rag":
                            st.markdown('<div class="badge badge-rag">🎯 Route: Local Vector DB</div>', unsafe_allow_html=True)
                        elif route_used == "web":
                            st.markdown('<div class="badge badge-web">🌐 Route: Web Search (MCP)</div>', unsafe_allow_html=True)
                        elif route_used == "both":
                            st.markdown('<div class="badge badge-both">⚡ Route: Hybrid (DB + Web)</div>', unsafe_allow_html=True)

                    # 2. Thinking process
                    reasoning, clean_content = parse_reasoning_and_answer(response)
                    if reasoning:
                        with st.expander("🧠 View Thinking Process"):
                            st.markdown(
                                f'<div class="thinking-container"><div class="thinking-title">Thinking Process</div><div class="thinking-content">{reasoning}</div></div>',
                                unsafe_allow_html=True
                            )

                    message_placeholder.write(clean_content)

                    # 3. Chunks
                    if docs:
                        with st.expander("🎯 View Retrieved Document Chunks"):
                            for idx, doc in enumerate(docs):
                                source = doc.get("metadata", {}).get("source", "Unknown Source") if doc.get("metadata") else "Unknown Source"
                                similarity = doc.get("similarity_score", 0.0)

                                st.markdown(f"""
                                <div class="citation-container">
                                    <div class="citation-header">
                                        <span>📄 Source {idx+1}: {source}</span>
                                        <span style="color: #1a73e8; font-weight:700;">Match: {similarity*100:.1f}%</span>
                                    </div>
                                    <div class="citation-body">"{doc['text']}"</div>
                                </div>
                                """, unsafe_allow_html=True)

                    # 4. Web search results
                    if web_results:
                        with st.expander("🌐 View Web Search References (MCP)"):
                            st.write(web_results)

                    # Append assistant response, documents, and routing details to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "documents": docs,
                        "route_used": route_used,
                        "web_results": web_results
                    })
                else:
                    try:
                        err_detail = r.json().get('detail', 'Unknown API Error')
                    except Exception:
                        err_detail = r.text
                    st.error(f"API Error: {err_detail}")
            except Exception as e:
                st.error(f"Pipeline error: {str(e)}")

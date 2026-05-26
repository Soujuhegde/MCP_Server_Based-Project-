import streamlit as st
import sys
from pathlib import Path
import os
import requests

# Backend API server URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


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
    layout="centered", # Centered layout is standard for ChatGPT
    initial_sidebar_state="expanded"
)

# Header
st.title("💬 EBook AI Navigator")
st.caption("A clean, minimalist RAG chatbot designed to query your documents like ChatGPT.")

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

# Setup Sidebar
with st.sidebar:
    st.markdown("### ⚙️ System Settings")
    st.caption(f"**Embedding Model:** `{embedding_model}`")
    st.caption(f"**LLM Engine:** `{llm_engine}`")
    st.caption(f"**Database:** `{database}`")
    st.markdown("---")
    
    # Dynamic Ingestion
    st.markdown("### 📥 Upload Documents")
    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT file to your knowledge base:",
        type=["pdf", "txt"]
    )
    
    if uploaded_file is not None:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if file_key not in st.session_state.ingested_files:
            with st.spinner("💾 Uploading and indexing document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    r = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=120)
                    
                    if r.status_code == 200:
                        st.session_state.ingested_files.add(file_key)
                        st.success(f"✅ Indexed '{uploaded_file.name}'!")
                        st.session_state.messages = [] # Reset chat history on new file
                        st.rerun()
                    else:
                        try:
                            err_detail = r.json().get('detail', 'Unknown API Error')
                        except Exception:
                            err_detail = r.text
                        st.error(f"Failed to ingest: {err_detail}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {str(e)}")
                
    st.markdown("---")
    
    # Active Files Panel
    st.markdown("### 📄 Active Documents")
    try:
        r = requests.get(f"{BACKEND_URL}/files", timeout=5)
        if r.status_code == 200:
            files = r.json().get("files", [])
            if files:
                for f in files:
                    st.caption(f"📄 `{f}`")
            else:
                st.caption("No files inside `data/` yet.")
        else:
            st.caption("Unable to load active documents.")
    except Exception:
        st.caption("Backend offline.")
        
    st.markdown("---")
    
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.ingested_files = set()
        st.success("Conversation cleared!")
        st.rerun()

# Display rolling chat messages from session state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            reasoning, clean_content = parse_reasoning_and_answer(msg["content"])
            if reasoning:
                with st.expander("🧠 View Thinking Process"):
                    st.caption(reasoning)
            st.write(clean_content)
        else:
            st.write(msg["content"])
        # If assistant has retrieved docs, display them inside a clean expander
        if msg["role"] == "assistant" and "documents" in msg and msg["documents"]:
            with st.expander("🎯 View Retrieved Chunks"):
                for idx, doc in enumerate(msg["documents"]):
                    source = doc["metadata"]["source"] if doc.get("metadata") else "Unknown source"
                    st.markdown(f"**Source {idx+1}:** *\"{doc['text']}\"*  \n🏷️ `{source}`")

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
                    "chat_history": st.session_state.messages[:-1]
                }
                r = requests.post(f"{BACKEND_URL}/query", json=payload, timeout=60)
                
                if r.status_code == 200:
                    result = r.json()
                    response = result["answer"]
                    docs = result["documents"]
                    
                    # Render response
                    reasoning, clean_content = parse_reasoning_and_answer(response)
                    if reasoning:
                        with st.expander("🧠 View Thinking Process"):
                            st.caption(reasoning)
                    message_placeholder.write(clean_content)
                    
                    # Show retrieved chunks immediately below
                    if docs:
                        with st.expander("🎯 View Retrieved Chunks"):
                            for idx, doc in enumerate(docs):
                                source = doc["metadata"]["source"] if doc.get("metadata") else "Unknown source"
                                st.markdown(f"**Source {idx+1}:** *\"{doc['text']}\"*  \n🏷️ `{source}`")
                    
                    # Append assistant response and documents to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "documents": docs
                    })
                else:
                    try:
                        err_detail = r.json().get('detail', 'Unknown API Error')
                    except Exception:
                        err_detail = r.text
                    st.error(f"API Error: {err_detail}")
            except Exception as e:
                st.error(f"Pipeline error: {str(e)}")

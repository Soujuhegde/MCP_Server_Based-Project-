from sentence_transformers import SentenceTransformer
import chromadb
from . import config

# Lazy initialization to avoid heavy loading at import time
_model = None
_client = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _model

def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    return _client

def search_docs(query, n_results=5):
    """Searches ChromaDB collection and returns structured chunks with source metadata."""
    model = _get_model()
    client = _get_client()
    
    # Fetch collection dynamically to prevent stale reference issues
    collection = client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME)

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0] if results.get("metadatas") else [None] * len(documents)
    print("metadata:",metadatas)
    # Return structured dicts containing both text and source metadata
    return [{"text": doc, "metadata": meta} for doc, meta in zip(documents, metadatas)]
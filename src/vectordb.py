from sentence_transformers import SentenceTransformer
import chromadb
from typing import Tuple, List, Dict, Optional
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

def search_docs(query: str, n_results: int = 5) -> Tuple[List[Dict], Optional[float]]:
    """
    Searches ChromaDB collection and returns structured chunks with source metadata and similarity scores.
    
    Args:
        query: Search query string
        n_results: Number of results to retrieve
    
    Returns:
        Tuple of (documents_list, best_similarity_score)
        - documents_list: List of dicts with 'text', 'metadata', 'similarity_score' keys
        - best_similarity_score: Highest similarity score (0-1) or None if no results
    """
    model = _get_model()
    client = _get_client()
    
    # Fetch collection dynamically to prevent stale reference issues
    collection = client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME)

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]  # Explicitly request distances
    )

    documents = results["documents"][0] if results.get("documents") else []
    metadatas = results["metadatas"][0] if results.get("metadatas") else [None] * len(documents)
    distances = results["distances"][0] if results.get("distances") else [1.0] * len(documents)
    
    print("metadata:", metadatas)
    print("distances:", distances)
    
    # Convert distances to similarity scores (0-1 range)
    # ChromaDB returns Euclidean distances; convert to cosine-like similarity
    best_similarity_score = None
    result_docs = []
    
    for doc, meta, distance in zip(documents, metadatas, distances):
        # Convert distance to similarity score (lower distance = higher similarity)
        # For normalized embeddings: similarity = 1 - distance
        similarity_score = max(0.0, 1.0 - distance)
        
        if best_similarity_score is None or similarity_score > best_similarity_score:
            best_similarity_score = similarity_score
        
        result_docs.append({
            "text": doc,
            "metadata": meta,
            "similarity_score": similarity_score
        })
    
    return result_docs, best_similarity_score
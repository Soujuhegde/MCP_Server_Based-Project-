from openai import OpenAI
from typing import Optional, List, Dict, Any
from . import config
from .vectordb import search_docs
from .prompts import PROMPT_TEMPLATE, CHAT_PROMPT_TEMPLATE
from .orchestrator import generate_answer_with_orchestration

# Initialize OpenAI-compatible client pointed at Sarvam's base URL
client = OpenAI(
    api_key=config.SARVAM_API_KEY,
    base_url=config.SARVAM_BASE_URL
)

def generate_answer(
    query: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    user_instructions: Optional[str] = None
) -> Dict:
    """
    Main entry point for answer generation.
    Routes through orchestrator if enabled, otherwise uses standard RAG.
    
    Args:
        query: User's question
        chat_history: Conversation history for context
        user_instructions: Custom system instructions from user
    
    Returns:
        Dict with 'answer', 'documents', 'route_used' keys
    """
    
    # Step 1: Retrieve docs from RAG
    rag_docs, best_score = search_docs(query)
    
    # Step 2: Use orchestrator if enabled
    if config.ORCHESTRATOR_ENABLED:
        return generate_answer_with_orchestration(
            query=query,
            rag_docs=rag_docs,
            rag_score=best_score,
            chat_history=chat_history,
            user_instructions=user_instructions
        )
    
    # Step 3: Fallback to standard RAG if orchestrator disabled
    context = "\n\n".join([doc["text"] for doc in rag_docs])
    
    if chat_history and len(chat_history) > 0:
        history_str = ""
        for msg in chat_history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            history_str += f"{role}: {content}\n"
        
        prompt = CHAT_PROMPT_TEMPLATE.format(
            context=context,
            chat_history=history_str.strip(),
            question=query
        )
    else:
        prompt = PROMPT_TEMPLATE.format(
            context=context,
            question=query
        )
    
    if user_instructions and user_instructions.strip():
        prompt = f"User's custom instructions:\n{user_instructions}\n\n{prompt}"
    
    response = client.chat.completions.create(
        model=config.SARVAM_MODEL_NAME,
        temperature=config.SARVAM_TEMPERATURE,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return {
        "answer": response.choices[0].message.content,
        "documents": rag_docs,
        "route_used": "rag"
    }
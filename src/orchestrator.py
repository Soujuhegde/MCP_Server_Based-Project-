import json
import logging
from typing import Dict, List, Optional, Tuple
from duckduckgo_search import DDGS
from openai import OpenAI

from . import config
from .prompts import (
    ORCHESTRATOR_PROMPT,
    WEB_CONTEXT_TEMPLATE,
    COMBINED_CONTEXT_PROMPT
)

logger = logging.getLogger(__name__)

# Initialize Sarvam AI client for orchestrator routing
client = OpenAI(
    api_key=config.SARVAM_API_KEY,
    base_url=config.SARVAM_BASE_URL
)

def classify_query_intent(query: str) -> Dict[str, str]:
    """
    Uses LLM to classify if query should hit RAG, web, or both.
    Returns JSON with 'route' and 'reasoning' keys.
    """
    try:
        prompt = ORCHESTRATOR_PROMPT.format(query=query)
        response = client.chat.completions.create(
            model=config.SARVAM_MODEL_NAME,
            temperature=0.3,  # Lower temp for consistent routing
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            result = json.loads(response_text)
            route = result.get("route", "rag").lower()
            
            # Validate route value
            if route not in ["rag", "web", "both"]:
                route = "both"  # Default to both if invalid
                
            return {"route": route, "reasoning": result.get("reasoning", "")}
        except json.JSONDecodeError:
            # If LLM didn't return valid JSON, default to "both"
            logger.warning(f"Orchestrator returned invalid JSON: {response_text}")
            return {"route": "both", "reasoning": "Defaulted to both due to parse error"}
            
    except Exception as e:
        logger.error(f"Error in query classification: {str(e)}")
        return {"route": "both", "reasoning": "Defaulted to both due to API error"}


def search_duckduckgo(query: str, max_results: int = None) -> str:
    """
    Performs Yahoo/DuckDuckGo web search and formats results as a string.
    No API key required — uses public search endpoints.
    
    Args:
        query: Search query string
        max_results: Max number of results (default from config)
    
    Returns:
        Formatted string of search results
    """
    if max_results is None:
        max_results = config.DDGO_MAX_RESULTS
    
    results = []
    
    # Method 1: Try Yahoo Search first (extremely robust, rarely blocked, rich snippets)
    try:
        import requests
        from bs4 import BeautifulSoup
        import urllib.parse
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        url = "https://search.yahoo.com/search"
        params = {"q": query}
        
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            for div in soup.find_all('div', class_='algo')[:max_results]:
                a_tag = div.find('a')
                if not a_tag:
                    continue
                
                # Extract clean title from h3 or fall back to a text
                title_tag = div.find('h3')
                title = title_tag.get_text().strip() if title_tag else a_tag.get_text().strip()
                
                # Extract clean redirect-free href
                href = a_tag.get('href', '')
                if "/RU=" in href:
                    try:
                        real_url = href.split("/RU=")[1].split("/")[0]
                        href = urllib.parse.unquote(real_url)
                    except Exception:
                        pass
                
                # Extract snippet body
                snippet_tag = div.find('span', class_='fc-falcon') or div.find('div', class_='compText') or div.find('p')
                body = snippet_tag.get_text().strip() if snippet_tag else "No snippet available."
                
                if title and href:
                    results.append({"title": title, "body": body, "href": href})
    except Exception as e:
        logger.warning(f"Yahoo Search parsing error: {str(e)}")
        
    # Method 2: Fallback to DuckDuckGo HTML scraping
    if not results:
        try:
            import requests
            from bs4 import BeautifulSoup
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            url = "https://html.duckduckgo.com/html/"
            params = {"q": query}
            
            r = requests.post(url, data=params, headers=headers, timeout=10)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                for div in soup.find_all('div', class_='result')[:max_results]:
                    title_tag = div.find('a', class_='result__url')
                    snippet_tag = div.find('a', class_='result__snippet')
                    
                    if title_tag:
                        title = title_tag.get_text().strip()
                        href = title_tag['href']
                        if href.startswith("//duckduckgo.com/y.js"):
                            continue
                    else:
                        title = "No Title"
                        href = ""
                        
                    body = snippet_tag.get_text().strip() if snippet_tag else "No snippet available."
                    results.append({"title": title, "body": body, "href": href})
        except Exception as e:
            logger.warning(f"DuckDuckGo HTML fallback error: {str(e)}")
            
    # Method 3: Fallback to duckduckgo_search library
    if not results:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
        except Exception as e:
            logger.error(f"DuckDuckGo search library error: {str(e)}")
            
    if not results:
        return "No web results found for this query."
    
    # Format results for LLM prompt
    formatted = []
    for idx, result in enumerate(results, 1):
        title = result.get("title", "")
        body = result.get("body", "")
        href = result.get("href", "")
        formatted.append(f"[{idx}] {title}\n    {body}\n    Source: {href}\n")
        
    return "\n".join(formatted)


def generate_answer_with_orchestration(
    query: str,
    rag_docs: List[Dict],
    rag_score: Optional[float] = None,
    chat_history: Optional[List[Dict]] = None,
    user_instructions: Optional[str] = None
) -> Dict:
    """
    Routes query to RAG, web, or both based on orchestrator decision.
    Combines contexts and generates final answer.
    
    Args:
        query: User's question
        rag_docs: Retrieved RAG documents with similarity scores
        rag_score: Best RAG match similarity score
        chat_history: Conversation history for context
        user_instructions: Custom system instructions from user
    
    Returns:
        Dict with 'answer', 'documents', 'route_used', 'web_results' keys
    """
    
    # Step 1: Classify intent
    classification = classify_query_intent(query)
    route = classification["route"]
    
    # Step 2: Check RAG confidence threshold
    if route == "rag" and rag_score is not None:
        if rag_score < config.RAG_CONFIDENCE_THRESHOLD:
            logger.info(f"RAG confidence {rag_score:.2f} below threshold {config.RAG_CONFIDENCE_THRESHOLD}. Switching to web.")
            route = "web"
    
    rag_context = ""
    web_context = ""
    web_results = ""
    
    # Step 3: Fetch context based on route
    if route in ["rag", "both"]:
        rag_context = "\n\n".join([doc["text"] for doc in rag_docs]) if rag_docs else "No relevant documents found."
    
    if route in ["web", "both"]:
        web_results = search_duckduckgo(query)
        web_context = WEB_CONTEXT_TEMPLATE.format(query=query, results=web_results)
    
    # Step 4: Build final prompt
    if route == "rag":
        # Use standard RAG prompt
        from .prompts import PROMPT_TEMPLATE
        final_prompt = PROMPT_TEMPLATE.format(context=rag_context, question=query)
    elif route == "web":
        # Use web-only context
        final_prompt = f"""You are a helpful AI assistant. Answer the user's question using the provided web search results.

Web Search Results:
{web_context}

Question: {query}

Answer:"""
    else:  # both
        # Use combined context prompt
        final_prompt = COMBINED_CONTEXT_PROMPT.format(
            rag_context=rag_context if rag_context else "No relevant documents found.",
            web_context=web_context,
            question=query
        )
    
    # Step 5: Add chat history if available
    if chat_history and len(chat_history) > 0:
        history_str = "\n".join([
            f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}"
            for msg in chat_history
        ])
        final_prompt = f"Conversation history:\n{history_str}\n\n{final_prompt}"
    
    # Step 6: Add user instructions if provided
    if user_instructions and user_instructions.strip():
        final_prompt = f"User's custom instructions:\n{user_instructions}\n\n{final_prompt}"
    
    # Step 7: Call LLM for final answer
    try:
        response = client.chat.completions.create(
            model=config.SARVAM_MODEL_NAME,
            temperature=config.SARVAM_TEMPERATURE,
            messages=[{"role": "user", "content": final_prompt}]
        )
        answer = response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        answer = f"Error generating answer: {str(e)}"
    
    return {
        "answer": answer,
        "documents": rag_docs if route in ["rag", "both"] else [],
        "route_used": route,
        "web_results": web_results if route in ["web", "both"] else ""
    }
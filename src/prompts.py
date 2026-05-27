PROMPT_TEMPLATE = """
You are a helpful AI assistant.

Answer ONLY from the provided context. If the answer cannot be found in the context, say "I cannot find the answer in the ebook context."

Context:
{context}

Question:
{question}

Answer:
"""

CHAT_PROMPT_TEMPLATE = """
You are a helpful conversational AI assistant.

Answer the user's question ONLY from the provided context. If the answer cannot be found in the context, say "I cannot find the answer in the ebook context."

Context:
{context}

Current Chat History:
{chat_history}

Question:
{question}

Answer:
"""

ORCHESTRATOR_PROMPT = """Analyze the user's query and determine if it should be answered from:
1. RAG (internal documents/knowledge base) 
2. WEB (real-time internet search)
3. BOTH (combination of documents and web)

Respond in JSON format ONLY with no explanation:
{{"route": "rag" | "web" | "both", "reasoning": "brief explanation"}}

Query: {query}
"""

WEB_CONTEXT_TEMPLATE = """Web search results for '{query}':

{results}

---
These are real-time web results. Cross-reference with your knowledge base if available.
"""

COMBINED_CONTEXT_PROMPT = """You are a helpful AI assistant. 

You have access to two sources of information:
1. User's document database (RAG context)
2. Real-time web search results

Answer the user's question using the BEST and MOST ACCURATE information from BOTH sources.
Always cite sources where applicable.

RAG Context (from user documents):
{rag_context}

Web Context (real-time search):
{web_context}

Question: {question}

Answer:
"""
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
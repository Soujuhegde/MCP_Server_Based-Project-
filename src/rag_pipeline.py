from openai import OpenAI
from . import config
from .vectordb import search_docs
from .prompts import PROMPT_TEMPLATE, CHAT_PROMPT_TEMPLATE

# Initialize OpenAI-compatible client pointed at Sarvam's base URL
client = OpenAI(
    api_key=config.SARVAM_API_KEY,
    base_url=config.SARVAM_BASE_URL
)

def generate_answer(query, chat_history=None):
    """Retrieves context docs and queries LLM, incorporating dialog memory if provided."""
    # Retrieve structured docs (containing both text and metadata)
    docs = search_docs(query)
    
    # Extract only the text representation of each retrieved chunk for context
    context = "\n\n".join([doc["text"] for doc in docs])

    # If chat history memory is present, construct the conversational prompt
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
        # Fallback to standard context prompt if chat memory is empty
        prompt = PROMPT_TEMPLATE.format(
            context=context,
            question=query
        )

    response = client.chat.completions.create(
        model=config.SARVAM_MODEL_NAME,
        temperature=config.SARVAM_TEMPERATURE,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return {
        "answer": response.choices[0].message.content,
        "documents": docs
    }
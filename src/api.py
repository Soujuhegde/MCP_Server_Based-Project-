import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# Add project root to sys.path to ensure src imports work when running from anywhere
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rag_pipeline import generate_answer
from src.ingest import ingest_ebook
from src import config 

app = FastAPI(
    title="EBook AI Navigator Backend",
    description="A FastAPI REST API backend decoupled from the Streamlit frontend.",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    query: str
    chat_history: Optional[List[Dict[str, Any]]] = None

@app.get("/config")
def get_config():
    """Returns system settings configuration details."""
    try:
        return {
            "embedding_model": config.EMBEDDING_MODEL_NAME,
            "llm_engine": config.SARVAM_MODEL_NAME,
            "database": "ChromaDB (Persistent)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files")
def list_files():
    """Lists active document files (.txt, .pdf) in the data directory."""
    try:
        data_dir = Path(config.EBOOK_FILE_PATH).parent
        if not data_dir.exists():
            return {"files": []}
        files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.pdf', '.txt'))]
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Saves uploaded PDF or TXT file to the data directory and triggers the ingestion pipeline."""
    try:
        dest_dir = Path(config.EBOOK_FILE_PATH).parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / file.filename
        
        # Save uploaded file chunk by chunk to avoid loading entire file in memory
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Trigger dynamic ingestion pipeline
        ingest_ebook()
        
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest: {str(e)}")

@app.post("/query")
def query_rag(request: QueryRequest):
    """Processes queries by passing them to the RAG pipeline with conversation history."""
    try:
        result = generate_answer(request.query, chat_history=request.chat_history)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)

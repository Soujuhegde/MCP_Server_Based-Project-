import os
from pathlib import Path
from dotenv import load_dotenv

# Find the root directory of the project (parent of src/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file at the project root
load_dotenv(dotenv_path=ROOT_DIR / ".env")

# Embedding Settings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ChromaDB Settings
CHROMA_DB_PATH = str(ROOT_DIR / "chroma_db")
CHROMA_COLLECTION_NAME = "ebook_collection"

# Data Settings
EBOOK_FILE_PATH = str(ROOT_DIR / "data" / "ebook.txt")

# LLM / Sarvam AI Settings
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_BASE_URL = "https://api.sarvam.ai/v1"
SARVAM_MODEL_NAME = "sarvam-m"
SARVAM_TEMPERATURE = 0.7

# Orchestrator Settings
ORCHESTRATOR_ENABLED = os.getenv("ORCHESTRATOR_ENABLED", "true").lower() == "true"
RAG_CONFIDENCE_THRESHOLD = float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.5"))
DDGO_MAX_RESULTS = int(os.getenv("DDGO_MAX_RESULTS", "5"))
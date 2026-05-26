import os
from sentence_transformers import SentenceTransformer
import chromadb
import sys
from pathlib import Path

# Try importing splitter from new/old LangChain paths
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

from langchain_community.document_loaders import PyPDFLoader

# Support running directly as a script vs. importing as a package
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config
else:
    from . import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def ingest_ebook():
    """Reads eBook (TXT) or PDFs, chunks them semantically, embeds chunks, and stores them in ChromaDB."""
    print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    data_dir = Path(config.EBOOK_FILE_PATH).parent
    print(f"Scanning directory for documents: {data_dir}...")
    
    raw_texts = []
    sources = []
    
    # Verify if directory exists
    if not data_dir.exists():
        print(f"⚠️ Directory {data_dir} does not exist. Creating it...")
        data_dir.mkdir(parents=True, exist_ok=True)
        
    # Check all files in the data directory
    for file_name in os.listdir(data_dir):
        file_path = data_dir / file_name
        if not file_path.is_file():
            continue
            
        print(f"Processing: {file_name}...")
        
        if file_name.lower().endswith(".pdf"):
            try:
                loader = PyPDFLoader(str(file_path))
                pages = loader.load()
                # Extract text from each page
                for page_num, page in enumerate(pages):
                    raw_texts.append(page.page_content)
                    sources.append(f"{file_name} (Page {page_num + 1})")
                print(f"  Loaded {len(pages)} pages from {file_name}")
            except Exception as e:
                print(f"  ❌ Error loading PDF {file_name}: {str(e)}")
                
        elif file_name.lower().endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                raw_texts.append(text)
                sources.append(file_name)
                print(f"  Loaded plain text file {file_name}")
            except Exception as e:
                print(f"  ❌ Error loading TXT {file_name}: {str(e)}")

    if not raw_texts:
        print("No documents (.txt or .pdf) found to ingest.")
        return

    # Semantic Chunking
    print("Chunking documents using RecursiveCharacterTextSplitter...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len
    )
    
    chunks = []
    metadata_list = []
    
    for doc_text, source_name in zip(raw_texts, sources):
        split_chunks = splitter.split_text(doc_text)
        for i, chunk in enumerate(split_chunks):
            if chunk.strip():
                chunks.append(chunk.strip())
                metadata_list.append({
                    "source": source_name,
                    "chunk_index": i
                })

    print(f"Generated {len(chunks)} semantic chunks.")
    if not chunks:
        print("No chunks generated.")
        return

    print("Generating embeddings...")
    embeddings = model.encode(chunks).tolist()

    print(f"Initializing ChromaDB Persistent Client at: {config.CHROMA_DB_PATH}...")
    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)

    # Storing in ChromaDB
    print(f"Getting or creating collection: '{config.CHROMA_COLLECTION_NAME}'...")
    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME
    )

    # Clean old collection data to prevent stale results
    print("Clearing previous vector database collection...")
    try:
        client.delete_collection(name=config.CHROMA_COLLECTION_NAME)
        collection = client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME
        )
    except Exception:
        pass

    # Store documents in ChromaDB
    print(f"Storing {len(chunks)} chunks in ChromaDB...")
    ids = [str(i) for i in range(len(chunks))]
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadata_list,
        ids=ids
    )

    print("Data stored successfully in ChromaDB!")

if __name__ == "__main__":
    ingest_ebook()
import argparse
import sys
import subprocess
from pathlib import Path

# Support Unicode/Emoji outputs in all terminal environments (especially Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add the project root directory to Python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ingest import ingest_ebook
from src.rag_pipeline import generate_answer
from src import config
from src.api import app  # Expose FastAPI app for uvicorn main:app

def main():
    parser = argparse.ArgumentParser(
        description="Ebook Knowledge RAG System - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --ingest
  python main.py --query "What is Retrieval Augmented Generation?"
  python main.py --chat
  python main.py --serve
"""
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-i", "--ingest",
        action="store_true",
        help="Ingest the documents (PDFs & TXT) from data directory into ChromaDB."
    )
    group.add_argument(
        "-q", "--query",
        type=str,
        help="Query the RAG pipeline directly from the command line and display the response."
    )
    group.add_argument(
        "-c", "--chat",
        action="store_true",
        help="Launch an interactive rolling chat session inside your terminal."
    )
    group.add_argument(
        "-s", "--serve",
        action="store_true",
        help="Launch both the FastAPI backend and Streamlit frontend."
    )
    group.add_argument(
        "-a", "--api",
        action="store_true",
        help="Launch only the FastAPI backend API server."
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if args.ingest:
        print("🚀 Starting Data Ingestion Pipeline...")
        ingest_ebook()
        
    elif args.query:
        question = args.query.strip()
        if not question:
            print("⚠️ Error: Please provide a non-empty question.")
            sys.exit(1)
            
        print(f"🔍 Question: '{question}'")
        print("🧠 Analyzing database & querying LLM...")
        try:
            result = generate_answer(question)
            print("\n" + "="*50)
            print("💡 SYNTHESIZED ANSWER:")
            print("="*50)
            print(result["answer"])
            print("\n" + "="*50)
            print("🎯 RETRIEVED CITATIONS:")
            print("="*50)
            for idx, doc in enumerate(result["documents"]):
                text = doc["text"]
                source = doc["metadata"]["source"] if doc.get("metadata") else "Unknown source"
                print(f"[{idx + 1}] \"{text}\"\n    (Source: {source})\n")
            print("="*50)
        except Exception as e:
            print(f"❌ Error occurred: {str(e)}")
            print("Please ensure your .env has valid keys and you have ingested data first.")
            sys.exit(1)
            
    elif args.chat:
        print("💬 Starting Conversational CLI Session. Type 'exit' or 'quit' to end.\n")
        chat_history = []
        while True:
            try:
                question = input("🧑 User: ").strip()
                if not question:
                    continue
                if question.lower() in ["exit", "quit"]:
                    print("👋 Ending session. Goodbye!")
                    break
                    
                print("🧠 Thinking...")
                result = generate_answer(question, chat_history=chat_history)
                print(f"\n🤖 AI: {result['answer']}")
                print("-" * 50)
                
                # Show references nicely
                print("🎯 Citations:")
                for idx, doc in enumerate(result["documents"]):
                    source = doc["metadata"]["source"] if doc.get("metadata") else "Unknown source"
                    print(f"  [{idx+1}] \"{doc['text'][:120]}...\" ({source})")
                print("="*50 + "\n")
                
                # Append to rolling chat history memory
                chat_history.append({"role": "user", "content": question})
                chat_history.append({"role": "assistant", "content": result['answer']})
            except KeyboardInterrupt:
                print("\n👋 Ending session. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error occurred: {str(e)}\n")
            
    elif args.api:
        print("🚀 Launching FastAPI Backend API Server...")
        import uvicorn
        try:
            uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=False)
        except KeyboardInterrupt:
            print("\n👋 FastAPI backend stopped.")
        except Exception as e:
            print(f"❌ Error starting API server: {str(e)}")
            sys.exit(1)

    elif args.serve:
        print("🌐 Launching Decoupled RAG System...")
        import time
        import requests
        
        # 1. Start uvicorn backend API server in a background subprocess
        api_cmd = [sys.executable, "-m", "uvicorn", "src.api:app", "--host", "127.0.0.1", "--port", "8000"]
        print("🚀 Starting FastAPI backend on http://127.0.0.1:8000...")
        
        api_process = subprocess.Popen(
            api_cmd, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
        
        # Wait for the backend API to become responsive
        print("⏳ Waiting for backend API to become responsive...")
        retries = 12
        backend_ready = False
        while retries > 0:
            try:
                res = requests.get("http://127.0.0.1:8000/config", timeout=1)
                if res.status_code == 200:
                    backend_ready = True
                    break
            except Exception:
                pass
            time.sleep(0.5)
            retries -= 1
            
        if not backend_ready:
            print("⚠️ Warning: Backend API did not respond in time. Proceeding anyway...")
        else:
            print("✅ Backend API is up and running!")
            
        # 2. Launch the Streamlit web application interface
        app_path = ROOT_DIR / "app" / "streamlit_app.py"
        print("🌐 Launching Streamlit interface...")
        try:
            subprocess.run(["streamlit", "run", str(app_path)], check=True)
        except KeyboardInterrupt:
            print("\n👋 Web application stopped.")
        except FileNotFoundError:
            print("❌ Error: 'streamlit' command not found. Please ensure it is installed in your virtual environment.")
            sys.exit(1)
        finally:
            # Clean up the background API server process when Streamlit exits
            print("🧹 Stopping FastAPI backend...")
            api_process.terminate()
            try:
                api_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                api_process.kill()
            print("👋 Goodbye!")

if __name__ == "__main__":
    main()

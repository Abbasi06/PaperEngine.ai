import os
import time
import subprocess
import urllib.request
import urllib.error
from contextlib import asynccontextmanager
from typing import Optional, List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from openai import OpenAI

# --- CONFIGURATION ---
# Adjust these paths to match your local setup
EXE_DIR = os.getenv("EXE_DIR", r"C:\Program Files\llama.cpp\build\bin\Release")
SERVER_EXE = "llama-server.exe"
SERVER_PATH = os.path.join(EXE_DIR, SERVER_EXE)

# Model Path (Using Qwen3-Embedding-8B or 0.6B)
# Ensure you have downloaded the GGUF version
MODEL_PATH = os.path.abspath(os.path.join("models", "Qwen3-Embedding-0.6B-f16.gguf"))

# --- PORT SETTINGS ---
# The External API will listen on 8007
# The Internal C++ Engine will listen on 8083 
# (Distinct ports to avoid conflicts with your Vision service)
ENGINE_PORT = 8083 
ENGINE_URL = os.getenv("EMBED_ENGINE_URL", f"http://localhost:{ENGINE_PORT}/v1")

# Global process handler
engine_process = None

# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle of the Qwen Embedding Engine.
    Starts llama-server with --embedding flag on startup, kills it on shutdown.
    """
    global engine_process
    print("\n🚀 [Lifespan] Starting Qwen Embedding Engine...")

    # 1. Verify Files Exist
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found at: {MODEL_PATH}")

    # 2. Start C++ Server (Only if binary exists)
    if os.path.exists(SERVER_PATH):
        cmd = [
            SERVER_PATH,
            "-m", MODEL_PATH,        # The Embedding Model
            "--embedding",           # Enable Embedding Mode (Disables generation)
            "--port", str(ENGINE_PORT),
            "-ngl", "99",            # Offload all layers to GPU
            "-c", "8192",            # 8k context is plenty for RAG chunks
            "--parallel", "4",       # Handle 4 concurrent embedding requests
            "--ubatch-size", "512"   # Batch size for processing
        ]
        
        # Start the process
        engine_process = subprocess.Popen(
            cmd, 
            cwd=EXE_DIR, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
    else:
        print(f"⚠️ {SERVER_EXE} not found. Assuming Docker/External Engine mode.")

    # 3. Health Check
    print(f"⏳ [Lifespan] Waiting for Embedding Engine on Port {ENGINE_PORT}...")
    engine_ready = False
    for i in range(45): 
        try:
            target_url = f"http://localhost:{ENGINE_PORT}/health"
            if not os.path.exists(SERVER_PATH):
                target_url = ENGINE_URL.replace("/v1", "/health").replace("localhost", "host.docker.internal")

            with urllib.request.urlopen(target_url) as response:
                if response.getcode() == 200:
                    print(f"✅ [Lifespan] Embedding Engine is Online!")
                    engine_ready = True
                    break
        except:
            time.sleep(1)
            
    if not engine_ready:
        if engine_process:
            print("❌ [Lifespan] Engine failed to start. Killing process.")
            engine_process.terminate()
            raise RuntimeError("Could not start Qwen Engine. Check paths and VRAM.")
        print("⚠️ Warning: Embedding Engine not reachable.")

    yield 

    # 4. Cleanup
    print("\n🛑 [Lifespan] Shutting down Embedding Engine...")
    if engine_process:
        engine_process.terminate()
        engine_process.wait()
        print("✅ [Lifespan] Engine stopped.")

# --- API APPLICATION ---
app = FastAPI(lifespan=lifespan, title="Embedding Service (Qwen3)")

# Internal Client
client = OpenAI(base_url=ENGINE_URL, api_key="local")

# Pydantic model for request validation
class EmbedRequest(BaseModel):
    texts: List[str] # Changed to accept a list of texts for batching
    instruction: Optional[str] = "Given a query, retrieve relevant documentation." 
    # Default instruction works well for general RAG

@app.post("/embed")
async def create_embedding(req: EmbedRequest):
    """
    Accepts a list of texts, applies Qwen instruction formatting, and returns vectors.
    """
    try:
        # 1. Format Input for Qwen Instruction-Awareness
        # This is critical for Qwen3 performance.
        formatted_inputs = [f"Instruct: {req.instruction}\nQuery: {text}" for text in req.texts]

        # 2. Call the Engine
        response = client.embeddings.create(
            model="qwen3-embedding", # Name doesn't matter strictly for llama.cpp, but good practice
            input=formatted_inputs
        )
        
        vectors = [item.embedding for item in response.data]
        dimensions = len(vectors[0]) if vectors else 0

        return {
            "object": "list",
            "model": "Qwen3-Embedding-8B",
            "dimensions": dimensions,
            "vectors": vectors
        }

    except Exception as e:
        print(f"❌ API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Embedding API on Port 8003...")
    uvicorn.run(app, host="0.0.0.0", port=8003)
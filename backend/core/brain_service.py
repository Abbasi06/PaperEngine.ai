import os
import time
import subprocess
import urllib.request
import urllib.error
from contextlib import asynccontextmanager
from fastapi import FastAPI
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
EXE_DIR = os.getenv("EXE_DIR", r"C:\Program Files\llama.cpp\build\bin\Release")
SERVER_EXE = "llama-server.exe"
SERVER_PATH = os.path.join(EXE_DIR, SERVER_EXE)

# Absolute path to where this script is
# In Docker, WORKDIR is /app, so models are at /app/models
MODEL_DIR = os.path.abspath("models")
FILENAME = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
REPO_ID = "bartowski/Llama-3.2-3B-Instruct-GGUF"
MODEL_PATH = os.path.join(MODEL_DIR, FILENAME)

ENGINE_PORT = 8081 
brain_process = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global brain_process
    print("\n🧠 [Lifespan] Starting Llama-3.2 Engine...")

    # 1. Ensure Model Exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print(f"⬇️ Downloading {FILENAME}...")
        hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=MODEL_DIR)

    # 2. Start C++ Server (Only if binary exists - Local Mode)
    if os.path.exists(SERVER_PATH):
        cmd = [
            SERVER_PATH,
            "-m", MODEL_PATH,
            "--port", str(ENGINE_PORT),
            "-ngl", "99",
            "-c", "8192"
        ]
        # Remove stdout=DEVNULL to see errors in terminal
        brain_process = subprocess.Popen(cmd)
    else:
        print(f"⚠️ {SERVER_EXE} not found. Assuming Docker/External Engine mode.")
        print(f"   Please ensure an LLM is running on {ENGINE_PORT} (or host.docker.internal).")

    # 3. Wait for Health Check
    print("⏳ Waiting for Engine to accept connections...")
    ready = False
    for _ in range(30):
        try:
            # We ping the INTERNAL port (8081)
            # In Docker, we might need to check host.docker.internal if we didn't spawn it
            target_url = f"http://localhost:{ENGINE_PORT}/health"
            if not os.path.exists(SERVER_PATH):
                target_url = f"http://host.docker.internal:{ENGINE_PORT}/health"
            
            urllib.request.urlopen(target_url)
            print("✅ Engine Ready!")
            ready = True
            break
        except:
            time.sleep(1)
            
    if not ready:
        if brain_process:
            print("❌ Engine failed to start.")
            brain_process.terminate()
            raise RuntimeError("Engine failed.")
        else:
            print("⚠️ Could not connect to external engine. Continuing anyway (Brain Service will be idle).")

    yield

    # 4. Cleanup
    if brain_process:
        print("🛑 Stopping Engine...")
        brain_process.terminate()

app = FastAPI(lifespan=lifespan)
# No endpoints needed! The app just needs to run to keep the lifespan active.

if __name__ == "__main__":
    import uvicorn
    # Run uvicorn programmatically
    uvicorn.run(app, host="0.0.0.0", port=8001)
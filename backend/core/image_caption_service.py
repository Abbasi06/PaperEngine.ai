import os
import time
import subprocess
import shutil
import base64
import urllib.request
import urllib.error
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from openai import OpenAI

# --- CONFIGURATION ---
# Adjust these paths if your installation varies
EXE_DIR = os.getenv("EXE_DIR", r"C:\Program Files\llama.cpp\build\bin\Release")
SERVER_EXE = "llama-server.exe"
SERVER_PATH = os.path.join(EXE_DIR, SERVER_EXE)

# Model Paths (Qwen2.5-VL)
MODEL_PATH = os.path.abspath(os.path.join("models", "Qwen2.5-VL-3B-Instruct-q4_k_m.gguf"))
MMPROJ_PATH = os.path.abspath(os.path.join("models", "Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf"))

# --- PORT SETTINGS ---
# The API will listen on 8006.
# The internal C++ Engine will listen on 8082.
# (We use 8082 so it doesn't fight with your Main Brain on 8081)
ENGINE_PORT = 8082 
ENGINE_URL = os.getenv("VISION_ENGINE_URL", f"http://localhost:{ENGINE_PORT}/v1")

# Global process handler
engine_process = None

# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle of the heavy C++ Vision Engine.
    Starts it when the API starts, kills it when the API stops.
    """
    global engine_process
    print("\n🚀 [Lifespan] Starting Qwen2.5-VL Engine (Vision)...")

    # 1. Verify Files Exist
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found at: {MODEL_PATH}")
    if not os.path.exists(MMPROJ_PATH):
        raise FileNotFoundError(f"❌ MMProj adapter not found at: {MMPROJ_PATH}")

    # 2. Start C++ Server (Only if binary exists)
    if os.path.exists(SERVER_PATH):
        cmd = [
            SERVER_PATH,
            "-m", MODEL_PATH,        # The Main Model
            "--mmproj", MMPROJ_PATH, # The Vision Adapter
            "--port", str(ENGINE_PORT),
            "-ngl", "99",            # Offload everything to GPU if possible
            "--ctx-size", "8192",    # Context size for images
            "--parallel", "1"
        ]
        
        # Start the process (Suppressing logs to keep terminal clean)
        engine_process = subprocess.Popen(
            cmd, 
            cwd=EXE_DIR, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
    else:
        print(f"⚠️ {SERVER_EXE} not found. Assuming Docker/External Engine mode.")

    # 3. Health Check (Wait for it to wake up)
    print(f"⏳ [Lifespan] Waiting for engine on Port {ENGINE_PORT}...")
    engine_ready = False
    for i in range(45): # Wait up to 45 seconds
        try:
            target_url = f"http://localhost:{ENGINE_PORT}/health"
            if not os.path.exists(SERVER_PATH):
                target_url = ENGINE_URL.replace("/v1", "/health").replace("localhost", "host.docker.internal")

            with urllib.request.urlopen(target_url) as response:
                if response.getcode() == 200:
                    print(f"✅ [Lifespan] Vision Engine is Online!")
                    engine_ready = True
                    break
        except:
            time.sleep(1)
            
    if not engine_ready:
        if engine_process:
            print("❌ [Lifespan] Engine failed to start. Killing process.")
            engine_process.terminate()
            raise RuntimeError("Could not start Qwen Engine. Check paths and VRAM.")
        print("⚠️ Warning: Vision Engine not reachable.")

    yield 

    # 4. Cleanup
    print("\n🛑 [Lifespan] Shutting down Vision Engine...")
    if engine_process:
        engine_process.terminate()
        engine_process.wait()
        print("✅ [Lifespan] Engine stopped.")

# --- API APPLICATION ---
app = FastAPI(lifespan=lifespan, title="Vision Service (Qwen-VL)")

# Internal Client to talk to the C++ Engine
client = OpenAI(base_url=ENGINE_URL, api_key="local")

@app.post("/analyze")
async def analyze_image(prompt: str, file: UploadFile = File(...)):
    """
    Accepts an image upload, converts it to Base64, and prompts Qwen-VL.
    """
    try:
        # 1. Read file into memory
        contents = await file.read()
        base64_img = base64.b64encode(contents).decode('utf-8')

        # 2. Call the Engine
        response = client.chat.completions.create(
            model="qwen2.5-vl",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]
                }
            ],
            max_tokens=512,
            temperature=0.1
        )
        return {"analysis": response.choices[0].message.content}

    except Exception as e:
        print(f"❌ API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Vision API on Port 8002...")
    # Strict Port Assignment
    uvicorn.run(app, host="0.0.0.0", port=8002)
# import os
# import shutil
# import json
# import requests
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from pydantic import BaseModel
# from openai import OpenAI
# from bs4 import BeautifulSoup
# import arxiv
# from typing import Optional
# from ddgs import DDGS
# import wikipedia

# # Internal Tools (Marker)
# from marker.converters.pdf import PdfConverter
# from marker.models import create_model_dict
# from marker.output import text_from_rendered
# from langchain_text_splitters import MarkdownHeaderTextSplitter

# # Import Prompts
# import prompts

# # --- MEMORY IMPORT (Robust) ---
# try:
#     from store import save_to_chroma
#     MEMORY_AVAILABLE = True
# except ImportError:
#     print("⚠️ Warning: store.py not found. Memory service will run in stub mode.")
#     MEMORY_AVAILABLE = False

# # --- CONFIGURATION ---
# app = FastAPI(title="Atomic AI Services (Optimized)")

# # Directories
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# TEMP_DIR = os.path.join(BASE_DIR, "temp_workspace")
# OUTPUT_IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")
# os.makedirs(TEMP_DIR, exist_ok=True)
# os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

# # Hardware: Keep Marker on CPU
# os.environ["TORCH_DEVICE"] = "CPU"

# # --- PORT CONFIGURATION ---
# VISION_API_URL = "http://localhost:8006/analyze" 
# BRAIN_URL = "http://localhost:8081/v1"
# EMBEDDING_API_URL = "http://localhost:8007/embed"

# client = OpenAI(base_url=BRAIN_URL, api_key="local")

# # --- DATA MODELS ---
# class TextPayload(BaseModel):
#     text: str
#     metadata: Optional[dict] = {}

# class GenerationPayload(BaseModel):
#     context: str
#     depth: str = "Medium" 
#     style: str = "Text"   
#     asset_type: str       

# class RankingPayload(BaseModel):
#     entities: list
#     query_context: str

# class SearchPayload(BaseModel):
#     query: str
#     max_results: int = 5
#     source_type: str = "web" # 'web' or 'arxiv'

# # ==========================================
# # MODULE 1: THE SENSES (Ingestion & Vision)
# # ==========================================

# @app.post("/tools/pdf_to_markdown")
# async def pdf_to_markdown(file: UploadFile = File(...)):
#     print(f"\n📄 [PDF Tool] Processing: {file.filename}...")
#     file_path = os.path.join(TEMP_DIR, file.filename)
    
#     try:
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
        
#         converter = PdfConverter(artifact_dict=create_model_dict())
#         rendered = converter(file_path)
#         text, _, images = text_from_rendered(rendered)
        
#         image_list = []
#         if images:
#             for filename, img in images.items():
#                 save_path = os.path.join(OUTPUT_IMAGE_DIR, filename)
#                 img.save(save_path)
#                 image_list.append(save_path)
        
#         print(f"   ✅ Success: Extracted {len(text)} chars and {len(image_list)} images.")
#         return {"markdown": text, "extracted_images": image_list}
        
#     except Exception as e:
#         print(f"   ❌ PDF Error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         if os.path.exists(file_path):
#             os.remove(file_path)

# @app.post("/tools/image_to_caption")
# async def image_to_caption(file: UploadFile = File(...)):
#     print(f"\n👁️ [Vision Tool] Analyzing: {file.filename}...")
#     try:
#         contents = await file.read()
#         files = {'file': (file.filename, contents, file.content_type)}
        
#         response = requests.post(
#             VISION_API_URL, 
#             params={'prompt': "Describe this detailedly. Focus on data trends."}, 
#             files=files, 
#             timeout=90
#         )
#         return response.json()
#     except requests.exceptions.ConnectionError:
#         print("   ❌ Error: Vision Service (Port 8006) unreachable.")
#         return {"analysis": "Error: Vision Service unreachable."}
#     except Exception as e:
#         print(f"   ❌ Vision Error: {e}")
#         return {"analysis": f"Error: {str(e)}"}

# # ==========================================
# # MODULE 2: THE DETECTIVE (Research)
# # ==========================================

# @app.post("/tools/extract_entities")
# async def extract_entities(payload: TextPayload):
#     print("\n🧠 [Extractor] Identifying Entities...")
#     prompt = prompts.KEYWORD_PROMPT.format(text=payload.text[:4000]) 
#     try:
#         response = client.chat.completions.create(
#             model="llama-3.2-3b-instruct",
#             messages=[{"role": "user", "content": prompt}],
#             response_format={"type": "json_object"},
#             temperature=0.1
#         )
#         return json.loads(response.choices[0].message.content)
#     except Exception as e:
#         print(f"   ❌ Extraction Error: {e}")
#         return {"entities": []}

# @app.post("/tools/rank_entities")
# async def rank_entities(payload: RankingPayload):
#     print(f"\n🔍 [Ranker] Optimization in progress...")
    
#     prompt = f"""
#     You are a Search Engine Optimization (SEO) expert.
    
#     Input Entities: {payload.entities}
#     User Intent: "{payload.query_context}"
    
#     Task 1: Identify the top 3-5 most technical and relevant entities.
#     Task 2: Construct a high-precision search query. 
    
#     Constraints for 'search_query':
#     - DO NOT just copy the User Intent.
#     - DO NOT list all keywords.
#     - Create a concise string (max 8 words) optimized for Google/Arxiv.
    
#     REQUIRED JSON FORMAT:
#     {{
#       "top_keywords": ["k1", "k2"],
#       "search_query": "OPTIMIZED_STRING"
#     }}
#     """
    
#     try:
#         response = client.chat.completions.create(
#             model="llama-3.2-3b-instruct",
#             messages=[{"role": "user", "content": prompt}],
#             response_format={"type": "json_object"},
#             temperature=0.1
#         )
#         return json.loads(response.choices[0].message.content)
#     except Exception as e:
#         print(f"   ❌ Ranker Error: {e}")
#         fallback_keys = payload.entities[:3]
#         short_intent = " ".join(payload.query_context.split()[:3])
#         return {
#             "top_keywords": fallback_keys,
#             "search_query": f"{short_intent} {' '.join(str(e) for e in fallback_keys)}"
#         }

# @app.post("/tools/search")
# async def search_external(payload: SearchPayload):
#     print(f"\n🕵️ [Detective] Starting Multi-Source Search: '{payload.query}'")
#     results = []
    
#     # 1. ACADEMIC SOURCE (Arxiv)
#     def run_arxiv(query, limit):
#         local_results = []
#         try:
#             print("   ...Checking Arxiv...")
#             client_arxiv = arxiv.Client()
#             search = arxiv.Search(query=query, max_results=limit, sort_by=arxiv.SortCriterion.Relevance)
#             for r in client_arxiv.results(search):
#                 local_results.append({
#                     "title": r.title, 
#                     "url": r.pdf_url, 
#                     "summary": r.summary[:200]+"...", 
#                     "source": "arxiv"
#                 })
#         except Exception as e:
#             print(f"   ❌ Arxiv Error: {e}")
#         return local_results

#     # 2. ENCYCLOPEDIC SOURCE (Wikipedia)
#     def run_wikipedia(query, limit):
#         local_results = []
#         try:
#             print("   ...Checking Wikipedia...")
#             titles = wikipedia.search(query, results=limit)
#             for title in titles:
#                 try:
#                     page = wikipedia.page(title, auto_suggest=False)
#                     local_results.append({
#                         "title": page.title, 
#                         "url": page.url, 
#                         "summary": page.summary[:200]+"...", 
#                         "source": "wikipedia"
#                     })
#                 except: continue
#         except Exception as e:
#             print(f"   ❌ Wikipedia Error: {e}")
#         return local_results

#     # 3. WEB SOURCE (DuckDuckGo Lite)
#     def run_duckduckgo(query, limit):
#         local_results = []
#         try:
#             print("   ...Checking DuckDuckGo (Lite)...")
#             with DDGS() as ddgs:
#                 ddg_gen = ddgs.text(query, max_results=limit+2, backend='lite')
#                 for r in ddg_gen:
#                     if len(local_results) >= limit: break
#                     local_results.append({
#                         "title": r.get('title', 'No Title'), 
#                         "url": r.get('href', '#'), 
#                         "summary": r.get('body', '')[:200]+"...", 
#                         "source": "web_search"
#                     })
#             if len(local_results) == 0:
#                 print("   ⚠️ DuckDuckGo returned 0 results.")
#             else:
#                 print(f"   ✅ Found {len(local_results)} web results.")
#         except Exception as e:
#             print(f"   ❌ DuckDuckGo Error: {e}")
#         return local_results

#     # EXECUTION CONTROLLER
#     if payload.source_type == "arxiv":
#         results.extend(run_arxiv(payload.query, payload.max_results))
#     else:
#         # Combined Web Search
#         limit = max(2, payload.max_results // 2) 
#         results.extend(run_wikipedia(payload.query, limit))
#         results.extend(run_arxiv(payload.query, limit))
#         # FIX: Renamed run_blogs -> run_duckduckgo to match definition
#         results.extend(run_duckduckgo(payload.query, limit))

#     print(f"   ✅ Combined Search Complete. Found {len(results)} total results.")
#     return {"results": results}

# @app.post("/tools/fetch_content")
# async def fetch_content(payload: TextPayload):
#     print(f"\n📖 [Reader] Fetching content from URL...")
#     url = payload.text
    
#     # STEALTH HEADERS (Mimics Chrome on Windows to reduce blocks)
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
#         "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
#         "Accept-Language": "en-US,en;q=0.5",
#         "Referer": "https://www.google.com/",
#         "Connection": "keep-alive",
#         "Upgrade-Insecure-Requests": "1"
#     }

#     try:
#         # ==================================================
#         # A. PDF HANDLING (Robust for Arxiv)
#         # ==================================================
#         if url.endswith(".pdf") or "arxiv.org/pdf" in url:
#             print("   👉 Detected PDF URL. Downloading...")
            
#             # 1. Download the PDF
#             resp = requests.get(url, headers=headers, timeout=30)
            
#             # --- [CRITICAL FIX START] ---
#             # This prevents the server from crashing if the PDF is blocked
#             if resp.status_code == 403:
#                 print("   ⚠️ Access Denied (PDF Source Blocked).")
#                 return {
#                     "error": "403_FORBIDDEN", 
#                     "detail": "PDF source blocked access. Agent should skip this link."
#                 }
            
#             if resp.status_code != 200:
#                 return {"error": f"Status: {resp.status_code}"}
#             # --- [CRITICAL FIX END] ---

#             # 2. Save Temp File
#             temp_pdf_path = os.path.join(TEMP_DIR, "temp_download.pdf")
#             with open(temp_pdf_path, "wb") as f:
#                 f.write(resp.content)
            
#             # 3. Convert using Marker
#             try:
#                 converter = PdfConverter(artifact_dict=create_model_dict())
#                 rendered = converter(temp_pdf_path)
#                 text, _, _ = text_from_rendered(rendered)
#                 print(f"   ✅ PDF Converted: {len(text)} chars")
#                 return {"content": text}
#             finally:
#                 # Always clean up the temp file
#                 if os.path.exists(temp_pdf_path):
#                     os.remove(temp_pdf_path)

#         # ==================================================
#         # B. WEB SCRAPING (HTML)
#         # ==================================================
#         else:
#             print("   👉 Detected Website. Scraping...")
#             resp = requests.get(url, headers=headers, timeout=15)
            
#             # Handle "Access Denied" gracefully
#             if resp.status_code == 403:
#                 print("   ⚠️ Access Denied (Bot Protection).")
#                 return {
#                     "error": "403_FORBIDDEN", 
#                     "detail": "Site blocked the scraper. Agent should try a different source."
#                 }
            
#             if resp.status_code != 200:
#                 return {"error": f"Status: {resp.status_code}"}
            
#             # Clean HTML
#             soup = BeautifulSoup(resp.content, 'html.parser')
#             for script in soup(["script", "style", "nav", "footer", "iframe", "header"]):
#                 script.extract()    
            
#             text = soup.get_text()
#             # Collapse multiple spaces/newlines
#             clean_text = ' '.join(text.split())
#             print(f"   ✅ Web Scraped: {len(clean_text)} chars")
#             return {"content": clean_text[:20000]}

#     except Exception as e:
#         print(f"   ❌ Reader Error: {e}")
#         return {"error": str(e)}



# # ==========================================
# # MODULE 3: THE BRAIN (Generation)
# # ==========================================

# @app.post("/generate/asset")
# async def generate_asset(req: GenerationPayload):
#     print(f"\n🎨 [Generator] Request for: {req.asset_type}")

#     # --- INTERNAL HELPER FUNCTION ---
#     # This prevents code duplication so we can call it once or 4 times
#     def run_llm_generation(target_type):
#         if target_type == "quiz":
#             prompt_template = prompts.QUIZ_PROMPT; json_mode = True
#         elif target_type == "flashcards":
#             prompt_template = prompts.FLASHCARD_PROMPT; json_mode = True
#         elif target_type == "mindmap":
#             prompt_template = prompts.MINDMAP_PROMPT; json_mode = False
#         elif target_type == "summary":
#             prompt_template = prompts.SUMMARY_PROMPT; json_mode = False
#         else:
#             raise ValueError(f"Unknown asset type: {target_type}")

#         final_prompt = prompt_template.format(context=req.context, depth=req.depth, style=req.style)
        
#         kwargs = {
#             "model": "llama-3.2-3b-instruct", 
#             "messages": [{"role": "user", "content": final_prompt}], 
#             "temperature": 0.3
#         }
#         if json_mode: 
#             kwargs["response_format"] = {"type": "json_object"}
            
#         response = client.chat.completions.create(**kwargs)
#         content = response.choices[0].message.content
        
#         # Return parsed JSON for structured data, or raw string for text
#         return json.loads(content) if json_mode else content

#     # --- MAIN LOGIC ---
#     try:
#         # Scenario A: Generate EVERYTHING
#         if req.asset_type == "all":
#             results = {}
#             # List of assets to generate
#             # We run them sequentially to be kind to your local GPU/CPU
#             tasks = ["summary", "quiz", "flashcards", "mindmap"]
            
#             for task in tasks:
#                 print(f"   ...generating {task}...")
#                 results[task] = run_llm_generation(task)
            
#             print("   ✅ All assets generated successfully.")
#             return results

#         # Scenario B: Generate SINGLE Asset
#         else:
#             output = run_llm_generation(req.asset_type)
#             # Maintain backward compatibility for non-JSON outputs
#             is_json = req.asset_type in ["quiz", "flashcards"]
#             return output if is_json else {"content": output}

#     except Exception as e:
#         print(f"   ❌ Generation Error: {e}")
#         raise HTTPException(500, detail=str(e))

# # ==========================================
# # MODULE 4: MEMORY (Storage)
# # ==========================================

# @app.post("/tools/get_embedding")
# async def get_embedding(payload: TextPayload):
#     """
#     Calls the Qwen Embedding Service (Port 8007) to vectorize text.
#     """
#     print(f"\n🧬 [Embedding] Vectorizing {len(payload.text)} chars...")
    
#     try:
#         # 1. Prepare Request to Microservice
#         # We assume the microservice expects {"text": "...", "instruction": "..."}
#         # You can customize the instruction based on usage if needed.
#         service_payload = {
#             "text": payload.text,
#             "instruction": "Represent this text for retrieval and search." 
#         }

#         # 2. Call the Service
#         response = requests.post(
#             EMBEDDING_API_URL, 
#             json=service_payload,
#             timeout=10 # Fast timeout
#         )
        
#         if response.status_code != 200:
#             raise HTTPException(status_code=response.status_code, detail="Embedding Service Failed")
            
#         data = response.json()
        
#         # 3. Return the vector
#         print(f"   ✅ Vector received: {data['dimensions']} dim")
#         return {
#             "vector": data['vector'],
#             "model": data['model']
#         }

#     except requests.exceptions.ConnectionError:
#         print("   ❌ Error: Embedding Service (Port 8007) is offline.")
#         raise HTTPException(status_code=503, detail="Embedding Service Unavailable")
#     except Exception as e:
#         print(f"   ❌ Embedding Error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     # Running on Port 8005
#     uvicorn.run(app, host="0.0.0.0", port=8005)

import os
import shutil
import json
import requests
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from bs4 import BeautifulSoup
import arxiv
from typing import Optional
from ddgs import DDGS
import wikipedia

# Internal Tools (Marker)
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# Import Prompts
from backend.prompts import prompts

# --- MEMORY CONFIGURATION ---
# We try to import ChromaDB. If it fails, memory features will be disabled.
try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: 'chromadb' not installed. Memory features will be disabled.")
    CHROMA_AVAILABLE = False

# --- CONFIGURATION ---
app = FastAPI(title="Atomic AI Services (Optimized)")

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp_workspace")
OUTPUT_IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")
MEMORY_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

# Hardware: Keep Marker on CPU
os.environ["TORCH_DEVICE"] = "CPU"

# --- PORT CONFIGURATION ---
VISION_API_URL = os.getenv("VISION_API_URL", "http://localhost:8002/analyze")
BRAIN_URL = os.getenv("BRAIN_URL", "http://localhost:8081/v1")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://localhost:8003/embed")

client = OpenAI(base_url=BRAIN_URL, api_key="local")

# --- INITIALIZE MEMORY ---
collection = None
if CHROMA_AVAILABLE:
    try:
        # Persistent storage in ./data/chroma_db
        chroma_client = chromadb.PersistentClient(path=MEMORY_DIR)
        # Create/Get collection. We manage embeddings manually, so no func needed.
        collection = chroma_client.get_or_create_collection(name="atomic_memory")
        print(f"🧠 [Memory] ChromaDB initialized at {MEMORY_DIR}")
    except Exception as e:
        print(f"⚠️ [Memory] Failed to init ChromaDB: {e}")

# --- DATA MODELS ---
class TextPayload(BaseModel):
    text: str
    metadata: Optional[dict] = {}

class GenerationPayload(BaseModel):
    context: str
    depth: str = "Medium" 
    style: str = "Text"   
    asset_type: str       

class RankingPayload(BaseModel):
    entities: list
    query_context: str

class SearchPayload(BaseModel):
    query: str
    max_results: int = 5
    source_type: str = "web" # 'web' or 'arxiv'

# --- HELPER FUNCTIONS ---
def clean_json_text(text):
    """Cleans LLM output to extract valid JSON."""
    try:
        # 1. Handle Markdown Code Blocks
        if "```" in text:
            parts = text.split("```")
            # Usually the content is in the second part (index 1)
            if len(parts) > 1:
                text = parts[1]
        
        text = text.strip()
        # Remove language identifier if present (e.g. "json")
        if text.lower().startswith("json"): 
            text = text[4:].strip()

        # 2. Find the outermost JSON structure (Object OR List)
        # We look for the first '{' or '[' and the last '}' or ']'
        start_idx = -1
        end_idx = -1
        
        # Check for Object
        first_brace = text.find("{")
        # Check for List
        first_bracket = text.find("[")
        
        # Determine which comes first
        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
            start_idx = first_brace
            end_idx = text.rfind("}")
        elif first_bracket != -1:
            start_idx = first_bracket
            end_idx = text.rfind("]")
            
        if start_idx != -1 and end_idx != -1:
            return text[start_idx : end_idx + 1]
            
        return text.strip()
    except: return text

def chunk_text(text, chunk_size=1500, overlap=100):
    """Splits text into manageable chunks for embedding."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]

# ==========================================
# MODULE 1: THE SENSES (Ingestion & Vision)
# ==========================================

@app.post("/tools/pdf_to_markdown")
async def pdf_to_markdown(file: UploadFile = File(...)):
    print(f"\n📄 [PDF Tool] Processing: {file.filename}...")
    file_path = os.path.join(TEMP_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(file_path)
        text, _, images = text_from_rendered(rendered)
        
        image_list = []
        if images:
            print(f"   👁️ Vision Enrichment: Analyzing {len(images)} images...")
            for filename, img in images.items():
                save_path = os.path.join(OUTPUT_IMAGE_DIR, filename)
                img.save(save_path)
                image_list.append(save_path)
                
                # --- Vision Enrichment ---
                try:
                    with open(save_path, "rb") as img_file:
                        files = {'file': (filename, img_file, "image/png")}
                        response = requests.post(
                            VISION_API_URL, 
                            params={'prompt': "Describe this scientific figure. Focus on data trends and labels."}, 
                            files=files, 
                            timeout=90
                        )
                    
                    if response.status_code == 200:
                        caption = response.json().get("analysis", "")
                        replacement = f"\n\n[[IMG_REF:{filename}]]\n> **Figure Analysis:** {caption}\n\n"
                        text = text.replace(f"![]({filename})", replacement)
                        print(f"      -> Captioned {filename}")
                except Exception as ve:
                    print(f"      ⚠️ Vision Enrichment Failed for {filename}: {ve}")
        
        print(f"   ✅ Success: Extracted {len(text)} chars and {len(image_list)} images.")
        return {"markdown": text, "extracted_images": image_list}
        
    except Exception as e:
        print(f"   ❌ PDF Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/tools/image_to_caption")
async def image_to_caption(file: UploadFile = File(...)):
    print(f"\n👁️ [Vision Tool] Analyzing: {file.filename}...")
    try:
        contents = await file.read()
        files = {'file': (file.filename, contents, file.content_type)}
        
        response = requests.post(
            VISION_API_URL, 
            params={'prompt': "Describe this detailedly. Focus on data trends."}, 
            files=files, 
            timeout=90
        )
        return response.json()
    except requests.exceptions.ConnectionError:
        print("   ❌ Error: Vision Service (Port 8006) unreachable.")
        return {"analysis": "Error: Vision Service unreachable."}
    except Exception as e:
        print(f"   ❌ Vision Error: {e}")
        return {"analysis": f"Error: {str(e)}"}


# ==========================================
# MODULE 2: THE DETECTIVE (Research)
# ==========================================

@app.post("/tools/extract_entities")
async def extract_entities(payload: TextPayload):
    print("\n🧠 [Extractor] Identifying Entities...")
    prompt = prompts.KEYWORD_PROMPT.format(text=payload.text[:4000]) 
    try:
        response = client.chat.completions.create(
            model="llama-3.2-3b-instruct",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"   ❌ Extraction Error: {e}")
        return {"entities": []}

@app.post("/tools/rank_entities")
async def rank_entities(payload: RankingPayload):
    print(f"\n🔍 [Ranker] Optimization in progress...")
    
    prompt = f"""
    You are a Search Engine Optimization (SEO) expert.
    
    Input Entities: {payload.entities}
    User Intent: "{payload.query_context}"
    
    Task 1: Identify the top 3-5 most technical and relevant entities.
    Task 2: Construct a high-precision search query. 
    
    Constraints for 'search_query':
    - DO NOT just copy the User Intent.
    - DO NOT list all keywords.
    - Create a concise string (max 8 words) optimized for Google/Arxiv.
    
    REQUIRED JSON FORMAT:
    {{
      "top_keywords": ["k1", "k2"],
      "search_query": "OPTIMIZED_STRING"
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="llama-3.2-3b-instruct",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"   ❌ Ranker Error: {e}")
        fallback_keys = payload.entities[:3]
        short_intent = " ".join(payload.query_context.split()[:3])
        return {
            "top_keywords": fallback_keys,
            "search_query": f"{short_intent} {' '.join(str(e) for e in fallback_keys)}"
        }

@app.post("/tools/search")
async def search_external(payload: SearchPayload):
    print(f"\n🕵️ [Detective] Starting Multi-Source Search: '{payload.query}'")
    results = []
    
    # 1. ACADEMIC SOURCE (Arxiv)
    def run_arxiv(query, limit):
        local_results = []
        try:
            print("   ...Checking Arxiv...")
            client_arxiv = arxiv.Client()
            search = arxiv.Search(query=query, max_results=limit, sort_by=arxiv.SortCriterion.Relevance)
            for r in client_arxiv.results(search):
                local_results.append({
                    "title": r.title, 
                    "url": r.pdf_url, 
                    "summary": r.summary[:200]+"...", 
                    "source": "arxiv"
                })
        except Exception as e:
            print(f"   ❌ Arxiv Error: {e}")
        return local_results

    # 2. ENCYCLOPEDIC SOURCE (Wikipedia)
    def run_wikipedia(query, limit):
        local_results = []
        try:
            print("   ...Checking Wikipedia...")
            titles = wikipedia.search(query, results=limit)
            for title in titles:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    local_results.append({
                        "title": page.title, 
                        "url": page.url, 
                        "summary": page.summary[:200]+"...", 
                        "source": "wikipedia"
                    })
                except: continue
        except Exception as e:
            print(f"   ❌ Wikipedia Error: {e}")
        return local_results

    # 3. WEB SOURCE (DuckDuckGo Lite)
    def run_duckduckgo(query, limit):
        local_results = []
        try:
            print("   ...Checking DuckDuckGo (Lite)...")
            with DDGS() as ddgs:
                ddg_gen = ddgs.text(query, max_results=limit+2)
                for r in ddg_gen:
                    if len(local_results) >= limit: break
                    local_results.append({
                        "title": r.get('title', 'No Title'), 
                        "url": r.get('href', '#'), 
                        "summary": r.get('body', '')[:200]+"...", 
                        "source": "web_search"
                    })
            if len(local_results) == 0:
                print("   ⚠️ DuckDuckGo returned 0 results.")
            else:
                print(f"   ✅ Found {len(local_results)} web results.")
        except Exception as e:
            print(f"   ❌ DuckDuckGo Error: {e}")
        return local_results

    # EXECUTION CONTROLLER
    if payload.source_type == "arxiv":
        results.extend(run_arxiv(payload.query, payload.max_results))
    else:
        # Combined Web Search
        limit = max(2, payload.max_results // 2) 
        results.extend(run_wikipedia(payload.query, limit))
        results.extend(run_arxiv(payload.query, limit))
        results.extend(run_duckduckgo(payload.query, limit))

    print(f"   ✅ Combined Search Complete. Found {len(results)} total results.")
    return {"results": results}

@app.post("/tools/fetch_content")
async def fetch_content(payload: TextPayload):
    print(f"\n📖 [Reader] Fetching content from URL...")
    url = payload.text
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }

    try:
        if url.endswith(".pdf") or "arxiv.org/pdf" in url:
            print("   👉 Detected PDF URL. Downloading...")
            resp = requests.get(url, headers=headers, timeout=30)
            
            if resp.status_code == 403:
                print("   ⚠️ Access Denied (PDF Source Blocked).")
                return {"error": "403_FORBIDDEN", "detail": "PDF source blocked access."}
            
            if resp.status_code != 200:
                return {"error": f"Status: {resp.status_code}"}

            temp_pdf_path = os.path.join(TEMP_DIR, "temp_download.pdf")
            with open(temp_pdf_path, "wb") as f:
                f.write(resp.content)
            
            try:
                converter = PdfConverter(artifact_dict=create_model_dict())
                rendered = converter(temp_pdf_path)
                text, _, _ = text_from_rendered(rendered)
                print(f"   ✅ PDF Converted: {len(text)} chars")
                return {"content": text}
            finally:
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)

        else:
            print("   👉 Detected Website. Scraping...")
            resp = requests.get(url, headers=headers, timeout=15)
            
            if resp.status_code == 403:
                print("   ⚠️ Access Denied (Bot Protection).")
                return {"error": "403_FORBIDDEN", "detail": "Site blocked the scraper."}
            
            if resp.status_code != 200:
                return {"error": f"Status: {resp.status_code}"}
            
            soup = BeautifulSoup(resp.content, 'html.parser')
            for script in soup(["script", "style", "nav", "footer", "iframe", "header"]):
                script.extract()    
            
            text = soup.get_text()
            clean_text = ' '.join(text.split())
            print(f"   ✅ Web Scraped: {len(clean_text)} chars")
            return {"content": clean_text[:20000]}

    except Exception as e:
        print(f"   ❌ Reader Error: {e}")
        return {"error": str(e)}


# ==========================================
# MODULE 3: THE BRAIN (Generation)
# ==========================================

@app.post("/generate/asset")
async def generate_asset(req: GenerationPayload):
    print(f"\n🎨 [Generator] Request for: {req.asset_type}")

    def run_llm_generation(target_type):
        if target_type == "quiz":
            prompt_template = prompts.QUIZ_PROMPT; json_mode = True
        elif target_type == "flashcards":
            prompt_template = prompts.FLASHCARD_PROMPT; json_mode = True
        elif target_type == "mindmap":
            prompt_template = prompts.MINDMAP_PROMPT; json_mode = False
        elif target_type == "summary":
            prompt_template = prompts.SUMMARY_PROMPT; json_mode = False
        elif target_type == "ppt":
            prompt_template = prompts.PPT_PROMPT; json_mode = True
        elif target_type == "video_script":
            prompt_template = prompts.VIDEO_SCRIPT_PROMPT; json_mode = True
        elif target_type == "podcast_script":
            prompt_template = prompts.PODCAST_SCRIPT_PROMPT; json_mode = True
        else:
            raise ValueError(f"Unknown asset type: {target_type}")

        final_prompt = prompt_template.format(context=req.context, depth=req.depth, style=req.style)
        
        kwargs = {
            "model": "llama-3.2-3b-instruct", 
            "messages": [{"role": "user", "content": final_prompt}], 
            "temperature": 0.3
        }
        if json_mode: 
            kwargs["response_format"] = {"type": "json_object"}
            
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        
        if json_mode:
            try:
                cleaned_content = clean_json_text(content)
                if not cleaned_content:
                    raise ValueError("LLM returned empty content after cleaning.")
                return json.loads(cleaned_content)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"   ⚠️ JSON Parsing Failed for {target_type}. Error: {e}")
                print(f"   Raw LLM Output: {content}")
                # Let the outer try-except handle the HTTPException
                raise ValueError(f"Failed to parse JSON from LLM for {target_type}")
        return content

    try:
        if req.asset_type == "all":
            results = {}
            tasks = ["summary", "quiz", "flashcards", "mindmap", "ppt", "video_script", "podcast_script"]
            for task in tasks:
                try:
                    print(f"   ...generating {task}...")
                    results[task] = run_llm_generation(task)
                except Exception as e:
                    print(f"   ⚠️ Failed to generate {task}: {e}")
                    results[task] = {"error": "Generation failed", "details": str(e)}
            
            print("   ✅ All assets generated successfully.")
            return results
        else:
            output = run_llm_generation(req.asset_type)
            is_json = req.asset_type in ["quiz", "flashcards", "ppt", "video_script", "podcast_script"]
            return output if is_json else {"content": output}

    except Exception as e:
        print(f"   ❌ Generation Error: {e}")
        raise HTTPException(500, detail=str(e))


# ==========================================
# MODULE 4: MEMORY (Storage & Embeddings)
# ==========================================

@app.post("/tools/get_embedding")
async def get_embedding(payload: TextPayload):
    """
    Calls the Qwen Embedding Service (Port 8007) to vectorize text.
    """
    print(f"\n🧬 [Embedding] Vectorizing {len(payload.text)} chars...")
    
    try:
        service_payload = {
            "text": payload.text,
            "instruction": "Represent this text for retrieval and search." 
        }

        # Call the standalone Embedding Service
        response = requests.post(
            EMBEDDING_API_URL, 
            json=service_payload,
            timeout=60 # Increased from 10s to 60s to prevent timeouts during model load
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Embedding Service Failed")
            
        data = response.json()
        
        print(f"   ✅ Vector received: {data['dimensions']} dim")
        return {
            "vector": data['vector'],
            "model": data['model']
        }

    except requests.exceptions.ConnectionError:
        print("   ❌ Error: Embedding Service (Port 8007) is offline.")
        raise HTTPException(status_code=503, detail="Embedding Service Unavailable")
    except Exception as e:
        print(f"   ❌ Embedding Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/store_memory")
async def store_memory(payload: TextPayload):
    """
    1. Gets vector from Qwen.
    2. Stores text+vector in ChromaDB.
    """
    print(f"\n💾 [Memory] Storing {len(payload.text)} chars...")
    
    if not collection:
        raise HTTPException(status_code=503, detail="ChromaDB not available")

    try:
        # 1. Chunk the text for embedding
        chunks = chunk_text(payload.text)
        print(f"   ...Splitting into {len(chunks)} chunks for storage.")
        
        if not chunks:
            return {"status": "no_content", "ids": [], "chunk_count": 0}

        # --- BATCH EMBEDDING (EFFICIENCY FIX) ---
        print(f"   ...Batch embedding {len(chunks)} chunks via Embedding Service...")
        service_payload = {
            "texts": chunks, # Send all chunks at once
            "instruction": "Represent this document for retrieval and search."
        }
        response = requests.post(
            EMBEDDING_API_URL,
            json=service_payload,
            timeout=180 # Generous timeout for batch embedding
        )
        response.raise_for_status()
        embedding_data = response.json()
        vectors = embedding_data['vectors']

        if len(vectors) != len(chunks):
            raise HTTPException(status_code=500, detail="Mismatch between chunks and returned vectors.")
        
        # --- BATCH SAVE TO CHROMADB ---
        doc_ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [(payload.metadata or {"source": "user_input", "chunk_index": i}) for i in range(len(chunks))]
        
        collection.add(ids=doc_ids, embeddings=vectors, documents=chunks, metadatas=metadatas)
        
        print(f"   ✅ Batch saved {len(doc_ids)} chunks to ChromaDB")
        return {
            "status": "stored", 
            "ids": doc_ids,
            "chunk_count": len(doc_ids)
        }

    except Exception as e:
        print(f"   ❌ Storage Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/search_memory")
async def search_memory(payload: SearchPayload):
    """
    Retrieves memories based on semantic similarity using Qwen vectors.
    """
    print(f"\n🧠 [Recall] Searching for: '{payload.query}'")
    if not collection: return []

    try:
        # 1. Vectorize the Query
        query_payload = TextPayload(text=payload.query)
        vector_resp = await get_embedding(query_payload)
        query_vector = vector_resp['vector']

        # 2. Search Chroma
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=payload.max_results,
            include=["documents", "metadatas", "distances"]
        )

        # 3. Format Results
        formatted_results = []
        if results['ids']:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "score": results['distances'][0][i] 
                })
        
        return formatted_results

    except Exception as e:
        print(f"   ❌ Search Error: {e}")
        return []

if __name__ == "__main__":
    import uvicorn
    print("🚀 Atomic Services running on Port 8004...")
    uvicorn.run(app, host="0.0.0.0", port=8004)

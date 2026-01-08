# import os
# import shutil
# import uvicorn
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from pydantic import BaseModel
# from typing import Optional
# from fastapi.middleware.cors import CORSMiddleware
# from agent_graph import app as agent_app

# app = FastAPI(title="PaperEngine Gateway")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# SESSIONS = {}

# class ChatRequest(BaseModel):
#     session_id: str
#     message: str
#     style_pref: Optional[str] = None # 'Reading', 'Watching', etc.
#     depth_pref: Optional[str] = None # 'Layman', 'Researcher'

# class ChatResponse(BaseModel):
#     response: str
#     debug_info: dict

# @app.post("/chat", response_model=ChatResponse)
# async def chat_endpoint(req: ChatRequest):
#     sid = req.session_id
    
#     if sid not in SESSIONS:
#         SESSIONS[sid] = {"messages": [], "file_path": None}
    
#     state = SESSIONS[sid]
    
#     # Store Preferences if provided
#     if req.style_pref: state["learning_style"] = req.style_pref
#     if req.depth_pref: state["depth_level"] = req.depth_pref

#     state["messages"].append({"role": "user", "content": req.message})
    
#     try:
#         final_state = agent_app.invoke(state)
#         # We handle "chat" intent vs "study" intent responses differently in real apps, 
#         # but here we just grab the last message or final_response
        
#         if final_state.get("final_response"):
#             bot_text = final_state["final_response"]
#         else:
#             bot_text = final_state["messages"][-1]["content"]

#         SESSIONS[sid] = final_state
        
#         return {
#             "response": bot_text,
#             "debug_info": {
#                 "intent": final_state.get("user_intent"),
#                 "style": final_state.get("learning_style"),
#                 "plan": final_state.get("execution_plan")
#             }
#         }
#     except Exception as e:
#         print(f"Error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/upload")
# async def upload_file(session_id: str, file: UploadFile = File(...)):
#     os.makedirs("uploads", exist_ok=True)
#     path = f"uploads/{file.filename}"
#     with open(path, "wb") as f:
#         shutil.copyfileobj(file.file, f)
        
#     if session_id not in SESSIONS: SESSIONS[session_id] = {"messages": []}
#     SESSIONS[session_id]["file_path"] = path
    
#     return {"status": "ok"}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


import os
import shutil
import uvicorn
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from backend.core.agent_graph import app as agent_app

app = FastAPI(title="PaperEngine Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str
    style_pref: Optional[str] = None
    depth_pref: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    artifacts: Optional[List[Dict[str, Any]]] = [] # New Field for Right Panel
    debug_info: dict

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    sid = req.session_id
    if sid not in SESSIONS: SESSIONS[sid] = {"messages": [], "file_path": None}
    
    state = SESSIONS[sid]
    if req.style_pref: state["learning_style"] = req.style_pref
    if req.depth_pref: state["depth_level"] = req.depth_pref
    state["messages"].append({"role": "user", "content": req.message})
    
    try:
        final_state = agent_app.invoke(state)
        
        # Extract Logic
        bot_text = final_state.get("final_response") or final_state["messages"][-1]["content"]
        artifacts = final_state.get("generated_artifacts", []) # Extract the list

        SESSIONS[sid] = final_state
        
        return {
            "response": bot_text,
            "artifacts": artifacts, # Send to frontend
            "debug_info": {
                "intent": final_state.get("user_intent"),
                "style": final_state.get("learning_style")
            }
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat_stream")
async def chat_stream(req: ChatRequest):
    """
    Streaming endpoint for Real-Time Progress updates.
    """
    sid = req.session_id
    if sid not in SESSIONS: 
        SESSIONS[sid] = {"messages": [], "file_path": None}
    
    # 1. Prepare State from Session
    initial_state = {
        "messages": SESSIONS[sid]["messages"] + [{"role": "user", "content": req.message}],
        "file_path": SESSIONS[sid].get("file_path"),
        "learning_style": req.style_pref,
        "depth_level": req.depth_pref,
        "is_info_complete": False 
    }

    if req.style_pref and req.depth_pref:
        initial_state["is_info_complete"] = True

    async def event_generator():
        final_response_text = ""
        try:
            # Stream events from the graph
            async for event in agent_app.astream(initial_state):
                for node_name, state_update in event.items():
                    if node_name == "planner":
                        plan = state_update.get("execution_plan", [])
                        yield f"data: {json.dumps({'type': 'plan', 'plan': plan})}\n\n"
                    elif node_name == "executor":
                        idx = state_update.get("current_step_index", 0)
                        artifacts = state_update.get("generated_artifacts", [])
                        latest_artifact = artifacts[-1] if artifacts else None
                        yield f"data: {json.dumps({'type': 'progress', 'completed_step': idx, 'artifact': latest_artifact})}\n\n"
                    elif node_name == "interviewer":
                        if state_update.get("final_response") == "SURVEY_REQUIRED":
                            yield f"data: {json.dumps({'type': 'control', 'action': 'SURVEY_REQUIRED'})}\n\n"
                    elif node_name == "chatbot":
                        msg = state_update["messages"][-1]["content"]
                        final_response_text = msg
                        yield f"data: {json.dumps({'type': 'response', 'content': msg})}\n\n"
            
            SESSIONS[sid]["messages"].append({"role": "user", "content": req.message})
            if final_response_text:
                SESSIONS[sid]["messages"].append({"role": "assistant", "content": final_response_text})
            yield "data: [DONE]\n\n"
        except Exception as e:
            print(f"Stream Error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/upload")
async def upload_file(session_id: str, file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{file.filename}"
    with open(path, "wb") as f: shutil.copyfileobj(file.file, f)
    if session_id not in SESSIONS: SESSIONS[session_id] = {"messages": []}
    SESSIONS[session_id]["file_path"] = path
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
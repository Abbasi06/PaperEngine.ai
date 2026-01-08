import os
import json
import requests
import operator
from typing import TypedDict, List, Annotated, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from openai import OpenAI
import logging

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
ENGINE_URL = os.getenv("ENGINE_URL", "http://localhost:8081/v1")
client = OpenAI(base_url=ENGINE_URL, api_key="local")
TOOLS_URL = os.getenv("TOOLS_URL", "http://localhost:8005")
MODEL_NAME = os.getenv("MODEL_NAME", "Llama-3.2-3B-Instruct-Q4_K_M.gguf")

def clean_json_text(text):
    try:
        # 1. Handle Markdown Code Blocks
        if "```" in text:
            parts = text.split("```")
            if len(parts) > 1:
                text = parts[1]
        
        text = text.strip()
        if text.lower().startswith("json"): 
            text = text[4:].strip()

        # 2. Find the outermost JSON structure
        start_idx = -1
        end_idx = -1
        
        first_brace = text.find("{")
        if first_brace != -1:
            start_idx = first_brace
            end_idx = text.rfind("}")
            
        if start_idx != -1 and end_idx != -1:
            return text[start_idx : end_idx + 1]
            
        return text.strip()
    except: return text

# --- STATE ---
class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add] 
    file_path: Optional[str]   
    
    # Metadata
    user_intent: Optional[str]       
    learning_style: Optional[str]    
    depth_level: Optional[str]       
    requested_outputs: Optional[List[str]] 
    generation_instruction: Optional[str]
    current_step_index: int # Track progress
    
    # Content
    context_text: Optional[str] 
    execution_plan: Optional[List[str]]
    
    # Outputs
    final_response: Optional[str] 
    generated_artifacts: Annotated[List[Dict[str, Any]], operator.add] # Accumulate artifacts
    is_info_complete: bool

# --- NODES ---


def chatbot_node(state: AgentState):
    """
    Context-Aware Chatbot:
    Uses the LLM to generate a natural response that acknowledges the user's input
    but pivots the conversation towards the Agent's capabilities (Study/Research).
    """
    logger.info("💬 CHAT: Generating context-aware response...")
    
    # 1. Get recent conversation history (Limit to last 5 to keep context fresh)
    recent_messages = state["messages"][-5:]
    
    # 2. Define the Persona & Goal
    system_prompt = """
    You are PaperEngine, an enthusiastic and helpful AI Research Assistant.
    
    YOUR GOAL:
    Engage in natural conversation, but ALWAYS gently steer the user towards:
    1. Uploading a document ("Study" mode).
    2. Defining a topic to explore ("Research" mode).
    
    GUIDELINES:
    - If the user says "Hi" or asks how you are, be warm and welcoming, then ask what they want to learn.
    - If the user engages in small talk, reply politely but ask: "What topic would you like to study or research today?"
    - Keep responses short (under 2 sentences).
    - Do NOT hallucinate capabilities (like booking flights). You only do Research and Study.
    """
    
    # 3. Call the Brain
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                *recent_messages
            ],
            max_tokens=150,
            temperature=0.7 # Slightly higher temp for natural conversation
        )
        
        bot_reply = response.choices[0].message.content
        logger.info(f"   💬 Chat Output: {bot_reply}")
        return {"messages": [{"role": "assistant", "content": bot_reply}]}
        
    except Exception as e:
        logger.error(f"   ❌ Chat Generation Error: {e}")
        # Fallback if LLM fails
        fallback = "I'm ready! What topic would you like to study or research today?"
        return {"messages": [{"role": "assistant", "content": fallback}]}

def intent_router(state: AgentState):
    logger.info(f"\n🧠 ROUTER: Analyzing Intent...")
    last_msg = state["messages"][-1]["content"]

    # Prompt designed to distinguish casual chat from work
    prompt = f"""
    Classify the user intent based on their message.
    
    Message: "{last_msg}"
    
    CATEGORIES:
    1. 'chat': Casual greetings (Hi, Hello, Good morning, How are you).
    2. 'work': User is asking to learn, study, research, explain, or summarize something.
    3. 'question': User is asking a specific question about the already processed content (e.g., "What is...", "Can you explain...").
    
    Return JSON: {{"intent": "chat" or "work"}}
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, max_tokens=100
        )
        data = json.loads(clean_json_text(response.choices[0].message.content))
        intent = data.get("intent", "chat").lower()
        logger.info(f"   🧠 Router Output: {intent}")
        return {"user_intent": intent}
    except: 
        logger.info(f"   🧠 Router Output: chat (fallback)")
        return {"user_intent": "chat"}

def interviewer_node(state: AgentState):
    logger.info(f"\n🎤 INTERVIEWER: Analyzing context for preferences...")
    
    user_msg = state["messages"][-1]["content"]
    current_style = state.get("learning_style")
    current_depth = state.get("depth_level")
    
    # --- 1. PRE-PROCESSING: Detect Direct Requests (Guardrail) ---
    direct_keywords = {
        "summary": "Summary", "summarize": "Summary",
        "quiz": "Quiz", "test": "Quiz",
        "flashcard": "Flashcards", "flashcards": "Flashcards",
        "mindmap": "MindMap", "mind map": "MindMap",
        "ppt": "PPT", "presentation": "PPT", "slides": "PPT",
        "video": "Video_Script", "youtube": "Video_Script",
        "podcast": "Podcast_Script", "audio": "Podcast_Script"
    }
    
    detected_assets = []
    lower_msg = user_msg.lower()
    for key, asset in direct_keywords.items():
        if key in lower_msg:
            if asset not in detected_assets:
                detected_assets.append(asset)

    # Check if we already have valid preferences to avoid loops
    has_existing_prefs = bool(current_style and current_depth)
    
    system_prompt = f"""
    You are the Interviewer. Analyze the user's request to determine if we have enough information to proceed or if we need to ask for preferences via a survey.

    CONTEXT:
    - User Message: "{user_msg}"
    - Current Style: {current_style}
    - Current Depth: {current_depth}
    - Detected Keywords (Hint): {detected_assets}

    TASKS:
    1. Detect if the user is explicitly asking for specific assets (e.g., "create a quiz", "summarize this"). If so, list them in 'requested_outputs'.
    2. Detect if the user is specifying a learning style (Reading, Listening, Watching, All) or depth (Layman, Researcher).
    3. If specific assets are requested, we do NOT need a survey. Set 'is_info_complete' to True.
    4. If no specific assets are requested AND we lack Style/Depth, we need a survey. Set 'is_info_complete' to False.
    5. If we have Style/Depth (either from context or history), map them to default outputs if no specific assets were requested.
       - Reading -> ["Summary", "Quiz", "Flashcards"]
       - Listening -> ["Summary", "Podcast_Script"]
       - Watching -> ["Summary", "Video_Script"]
       - All -> ["Summary", "Quiz", "Flashcards", "MindMap", "PPT", "Podcast_Script", "Video_Script"]

    Return JSON:
    {{
        "requested_outputs": ["Summary", ...],
        "learning_style": "Reading" | "Listening" | "Watching" | "All" | "Direct",
        "depth_level": "Layman" | "Researcher",
        "is_info_complete": true | false,
        "reasoning": "..."
    }}
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": system_prompt}],
            response_format={"type": "json_object"},
            max_tokens=300,
            temperature=0.1
        )
        data = json.loads(clean_json_text(response.choices[0].message.content))
        
        requested_outputs = data.get("requested_outputs", [])
        new_style = data.get("learning_style") or current_style
        new_depth = data.get("depth_level") or current_depth
        is_complete = data.get("is_info_complete", False)
        reasoning = data.get("reasoning", "")

        logger.info(f"   🤔 Interviewer Reasoning: {reasoning}")

        # --- ROBUSTNESS LOGIC ---
        
        # 1. Merge detected assets if LLM missed them
        for asset in detected_assets:
            if asset not in requested_outputs:
                requested_outputs.append(asset)

        # 2. If we have direct requests, we are complete.
        if requested_outputs:
             is_complete = True
             if not new_style: new_style = "Direct"
             if not new_depth: new_depth = "Layman"

        # 3. If we already had prefs, force complete (unless user explicitly resets)
        if has_existing_prefs:
            is_complete = True

        # 4. Programmatic Defaults: If style is known but outputs are empty, fill them.
        if is_complete and not requested_outputs and new_style:
            if new_style == "Reading": requested_outputs = ["Summary", "Quiz", "Flashcards"]
            elif new_style == "Listening": requested_outputs = ["Summary", "Podcast_Script"]
            elif new_style == "Watching": requested_outputs = ["Summary", "Video_Script"]
            elif new_style == "All": requested_outputs = ["Summary", "Quiz", "Flashcards", "MindMap", "PPT", "Podcast_Script", "Video_Script"]
            elif new_style == "Direct": requested_outputs = ["Summary"] # Fallback for Direct

        if not is_complete:
             logger.warning("   ⚠️ Info incomplete. Requesting Survey.")
             return {
                "final_response": "SURVEY_REQUIRED",
                "is_info_complete": False
            }
        
        logger.info(f"   ✅ Interviewer Output: {requested_outputs} (Style: {new_style})")
        return {
            "requested_outputs": requested_outputs,
            "learning_style": new_style,
            "depth_level": new_depth,
            "is_info_complete": True,
            "final_response": "" # CRITICAL FIX: Clear stale "SURVEY_REQUIRED"
        }

    except Exception as e:
        logger.error(f"   ❌ Interviewer LLM Error: {e}")
        
        # Fallback: If we have detected assets, proceed.
        if detected_assets:
             return {
                 "requested_outputs": detected_assets, 
                 "learning_style": "Direct", 
                 "depth_level": "Layman", 
                 "is_info_complete": True,
                 "final_response": "" # Clear stale state
             }

        # Fallback: If we have state, proceed with defaults.
        if current_style and current_depth:
             defaults = ["Summary"]
             if current_style == "Reading": defaults = ["Summary", "Quiz", "Flashcards"]
             elif current_style == "Listening": defaults = ["Summary", "Podcast_Script"]
             elif current_style == "Watching": defaults = ["Summary", "Video_Script"]
             return {"requested_outputs": defaults, "is_info_complete": True, "final_response": ""}
             
        return {"final_response": "SURVEY_REQUIRED", "is_info_complete": False}

def planner_node(state: AgentState):
    logger.info(f"\n📐 PLANNER: Reasoning about execution...")
    
    # 1. Gather Context
    user_msg = state["messages"][-1]["content"]
    style = state.get("learning_style")
    initial_outputs = state.get("requested_outputs", [])
    has_file = bool(state.get("file_path"))
    
    # 2. LLM Planning (Chain of Thought)
    system_prompt = f"""
    You are the Planning Engine, Create a robust execution plan based on the user's request and context.
    
    CONTEXT:
    - User Request: "{user_msg}"
    - File Uploaded: {has_file}
    - Learning Style: {style} (Defaults: {initial_outputs})
    
    AVAILABLE TOOLS:
    - tool:ingest_pdf (Use ONLY if File Uploaded is True)
    - tool:research_topic (Use if File Uploaded is False)
    - tool:store_memory (Mandatory step after ingestion/research)
    - generate:summary
    - generate:quiz
    - generate:flashcards
    - generate:mindmap
    - generate:ppt
    - generate:video_script
    - generate:podcast_script
    
    TASK:
    1. Analyze the request. If the user asks for specific outputs (e.g. "just a quiz"), prioritize those over the style defaults.
    2. Determine the source: If a file is uploaded, you MUST start with 'tool:ingest_pdf'. If not, start with 'tool:research_topic'.
    3. Always include 'tool:store_memory' after the source tool.
    4. List the generation tools required.
    
    Return JSON:
    {{
      "reasoning": "Step-by-step logic...",
      "plan": ["tool:...", ...],
      "focus_instruction": "Specific guidance for generation..."
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, messages=[{"role": "user", "content": system_prompt}],
            response_format={"type": "json_object"}, max_tokens=500, temperature=0.1
        )
        data = json.loads(clean_json_text(response.choices[0].message.content))
        
        plan = data.get("plan", [])
        reasoning = data.get("reasoning", "No reasoning provided.")
        focus_instruction = data.get("focus_instruction", "Standard coverage")
        
        # Robustness Check: Ensure LLM returned a list
        if not isinstance(plan, list): 
            logger.warning("   ⚠️ Planner returned invalid plan format. Using defaults.")
            plan = []

        # Fallback if plan is empty (LLM might have failed to generate steps)
        if not plan:
            if has_file: plan.append("tool:ingest_pdf")
            else: plan.append("tool:research_topic")
            plan.append("tool:store_memory")
            for out in initial_outputs:
                plan.append(f"generate:{out.lower()}")

    except Exception as e:
        logger.error(f"   ⚠️ Planner LLM failed, using defaults: {e}")
        # Hard Fallback
        plan = []
        if has_file: plan.append("tool:ingest_pdf")
        else: plan.append("tool:research_topic")
        plan.append("tool:store_memory")
        for out in initial_outputs:
            plan.append(f"generate:{out.lower()}")
        reasoning = "Fallback due to error."
        focus_instruction = "Standard coverage"
        
    logger.info(f"   🤔 Reasoning: {reasoning}")
    logger.info(f"   📝 Plan: {plan}")
    logger.info(f"   🎯 Instruction: {focus_instruction}")
    
    # Extract requested outputs for the state (just for record keeping)
    refined_outputs = [p.split(":")[1] for p in plan if p.startswith("generate:")]

    logger.info(f"   📐 Planner Output (Execution Steps): {plan}")
        
    return {
        "execution_plan": plan, 
        "requested_outputs": refined_outputs,
        "generation_instruction": focus_instruction,
        "current_step_index": 0, # Initialize progress
        "generated_artifacts": [] # Initialize artifacts container
    }

def retriever_node(state: AgentState):
    """
    Retrieves relevant context from ChromaDB based on the user's question.
    """
    logger.info("\n🔍 RETRIEVER: Searching memory for relevant context...")
    user_question = state["messages"][-1]["content"]
    try:
        response = requests.post(
            f"{TOOLS_URL}/tools/search_memory",
            json={"query": user_question, "max_results": 5},
            timeout=30
        )
        response.raise_for_status()
        results = response.json()
        retrieved_docs = [item.get("content", "") for item in results]
        context = "\n\n---\n\n".join(retrieved_docs)
        logger.info(f"   ✅ Retrieved {len(retrieved_docs)} chunks from memory.")
        return {"context_text": context}
    except Exception as e:
        logger.error(f"   ❌ Retriever Error: {e}")
        return {"context_text": "Failed to retrieve context from memory."}

def rag_node(state: AgentState):
    """
    Generates an answer to the user's question using the retrieved context.
    """
    logger.info("\n💡 RAG: Generating answer from context...")
    user_question = state["messages"][-1]["content"]
    context = state.get("context_text", "No context available.")
    
    prompt = f"Context:\n{context}\n\nQuestion: {user_question}\n\nAnswer:"
    
    response = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=0.2, max_tokens=500)
    answer = response.choices[0].message.content
    return {"messages": [{"role": "assistant", "content": answer}]}

def executor_node(state: AgentState):
    """
    The Execution Engine.
    Executes ONE step of the plan at a time to allow for real-time progress updates.
    """
    plan = state.get("execution_plan", [])
    idx = state.get("current_step_index", 0)
    
    if idx >= len(plan):
        return {"current_step_index": idx} # Should be caught by conditional edge, but safety first

    step = plan[idx]
    logger.info(f"\n⚙️ EXECUTOR: Step {idx+1}/{len(plan)} -> {step}")
    
    # Local state
    context = state.get("context_text", "")
    artifacts = []
    
    # Configuration
    depth = state.get("depth_level", "Medium")
    style = state.get("learning_style", "Text")
    instruction = state.get("generation_instruction", "")
    
    # --- 1. INGESTION ---
    if step == "tool:ingest_pdf":
        file_path = state.get("file_path")
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    files = {'file': (os.path.basename(file_path), f, 'application/pdf')}
                    response = requests.post(f"{TOOLS_URL}/tools/pdf_to_markdown", files=files, timeout=600)
                if response.status_code == 200:
                    context = response.json().get("markdown", "")
                    logger.info(f"      ✅ Ingested {len(context)} chars.")
                else:
                    logger.error("      ❌ Ingestion failed.")
            except Exception as e:
                logger.error(f"      ❌ Ingestion error: {e}")
        else:
            logger.warning("      ⚠️ No file found to ingest.")

    # --- 2. RESEARCH ---
    elif step == "tool:research_topic":
        query = state["messages"][-1]["content"]
        try:
            response = requests.post(f"{TOOLS_URL}/tools/search", json={"query": query}, timeout=30)
            if response.status_code == 200:
                context = str(response.json().get("results", ""))
                logger.info(f"      ✅ Research complete.")
            else:
                logger.error("      ❌ Research failed.")
        except Exception as e:
            logger.error(f"      ❌ Research error: {e}")

    # --- 3. MEMORY ---
    elif step == "tool:store_memory":
        if context:
            try:
                requests.post(
                    f"{TOOLS_URL}/tools/store_memory", 
                    json={"text": context, "metadata": {"source": "paper_engine"}},
                    timeout=300 # Increased timeout for large documents
                )
                logger.info("      ✅ Stored in memory.")
            except Exception as e:
                logger.error(f"      ❌ Memory store error: {e}")

    # --- 4. GENERATION ---
    elif step.startswith("generate:"):
        asset_type = step.split(":")[1]
        
        # Map to API asset types
        api_asset_type = asset_type
        if "video" in asset_type: api_asset_type = "video_script"
        elif "podcast" in asset_type: api_asset_type = "podcast_script"
        
        payload = {
            "context": context[:15000], 
            "depth": depth,
            "style": style,
            "asset_type": api_asset_type,
            "instruction": instruction
        }
        
        try:
            logger.info(f"      ... Generating {api_asset_type} ...")
            resp = requests.post(f"{TOOLS_URL}/generate/asset", json=payload, timeout=300)
            if resp.status_code == 200:
                result = resp.json()
                display_type = asset_type.replace("_", " ").title()
                if "Ppt" in display_type: display_type = "PPT"
                
                artifacts.append({ "type": display_type, "data": result })
                logger.info(f"      ✅ Generated {display_type}")
            else:
                logger.error(f"      ❌ Generation failed for {asset_type}")
        except Exception as e:
            logger.error(f"      ❌ Generation error: {e}")

    updates = {
        "context_text": context,
        "generated_artifacts": artifacts,
        "current_step_index": idx + 1 # Increment progress
    }
    
    # If this was the last step, set a completion message
    if idx + 1 >= len(plan):
        updates["final_response"] = "I've finished generating the requested materials. You can view them in the side panel."
        
    return updates

# --- GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("router", intent_router)
workflow.add_node("interviewer", interviewer_node)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("rag", rag_node)
workflow.add_node("chatbot", chatbot_node)

workflow.set_entry_point("router")

def route_logic(state):
    intent = state.get("user_intent")
    if intent == "chat": return "chatbot"
    if intent == "question": return "retriever"
    return "interviewer"

def interview_logic(state):
    # If Interviewer says SURVEY_REQUIRED (is_info_complete=False), STOP.
    if not state.get("is_info_complete"): return END
    return "planner"

def execution_logic(state):
    # Loop back to executor if there are more steps
    plan = state.get("execution_plan", [])
    idx = state.get("current_step_index", 0)
    if idx < len(plan): return "executor"
    return END

workflow.add_conditional_edges("router", route_logic)
workflow.add_conditional_edges("interviewer", interview_logic)
workflow.add_edge("planner", "executor")
workflow.add_edge("retriever", "rag")
workflow.add_edge("rag", END)
workflow.add_conditional_edges("executor", execution_logic)
workflow.add_edge("chatbot", END)

app = workflow.compile()
logger.info("✅ Agent Graph Initialized.")
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- Configuration (Must match training setup) ---
MODEL_NAME = "microsoft/phi-2"
ADAPTER_PATH = "./final_phi_json_adapter" 
# NOTE: The EOS token will be used as the PAD token during inference
MAX_GENERATION_LENGTH = 1024 

# --- 1. Load the Model and Adapters ---
def load_fine_tuned_model():
    """Loads the base Phi model, applies QLoRA adapters, and returns the pipeline."""
    
    # 1. Quantization Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # 2. Load Tokenizer and Set Padding Token (Crucial Fix)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # Set padding side for generation

    # 3. Load Base Model (4-bit)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # 4. Load and Merge LoRA Adapters
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH) 
    # Merge for potentially cleaner inference if VRAM allows, otherwise keep separate
    # model = model.merge_and_unload() 
    
    # Set model to evaluation mode
    model.eval()
    
    return model, tokenizer

# --- 2. Generation Function ---
def generate_ppt_json(raw_text: str, available_images: list, model, tokenizer) -> dict or None:
    """
    Takes document text and image list, generates the structured JSON output, 
    and robustly cleans the output string to handle extra text/contamination.
    """
    
    # 1. Construct the EXACT Instruction Template 
    instruction = (
        "You are a highly analytical Presentation Architect. Your sole task is to convert the provided document excerpt "
        "and image list into a structured, multi-slide JSON presentation outline. Return ONLY the JSON object. "
        "Do not include any explanations, markdown fences (like `json`), or commentary. The output must strictly "
        "adhere to the provided schema and its explicit constraints."
    )
    
    image_list_str = str(available_images)
    input_content = f"---INPUT TEXT---\n{raw_text}\n\n---AVAILABLE IMAGES---\n{image_list_str}"

    # The prompt must use the exact pattern trained: ### Output:\n
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_content}\n\n### Output:\n"
    
    # 2. Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    # 3. Generate the tokens
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_GENERATION_LENGTH,
            do_sample=True,      
            temperature=0.4, 
            top_p=0.9,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id 
        )
        
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # --- ROBUST JSON EXTRACTION LOGIC ---
    try:
        # 1. Find the starting point of the JSON after the prompt marker
        json_start_index = generated_text.find("### Output:")
        if json_start_index == -1:
            raise ValueError("Output marker not found.")
        
        # Start looking for the JSON string just after the marker
        json_string_candidate = generated_text[json_start_index:].split("### Output:")[-1].strip()
        
        # 2. Aggressive Cleanup of Markdown Fences
        # If the model adds ```json or ``` at the start/end, remove them
        json_string_candidate = json_string_candidate.strip()
        if json_string_candidate.startswith('```'):
            json_string_candidate = json_string_candidate[json_string_candidate.find('\n')+1:]
        if json_string_candidate.endswith('```'):
            json_string_candidate = json_string_candidate[:-3]
        
        # 3. Final extraction by finding the first '{' and the last '}'
        # This is the most reliable way to strip contamination
        first_brace = json_string_candidate.find('{')
        last_brace = json_string_candidate.rfind('}')
        
        if first_brace == -1 or last_brace == -1 or last_brace < first_brace:
            raise ValueError("JSON braces not found or malformed.")

        # Extract the content between the first { and the last } (inclusive)
        json_string = json_string_candidate[first_brace : last_brace + 1].strip()
        
        # 4. Attempt to parse the cleaned string
        return json.loads(json_string)
        
    except (json.JSONDecodeError, ValueError, IndexError, AttributeError) as e:
        print(f"--- FAILED TO PARSE JSON ---")
        print(f"Error: {e}")
        print(f"Raw Generated Text (Full Output):\n{generated_text}")
        print(f"Candidate JSON String (Attempted Fix):\n{json_string_candidate[:500]}...")
        return None

# --- 3. Execution Example ---
if __name__ == "__main__":
    # Simulate data coming from your Marker/Qwen services
    SAMPLE_RAW_TEXT = (
        "The shift towards remote work has made Zero Trust the standard security model. "
        "Unlike perimeter defenses, Zero Trust mandates strict, continuous verification for every "
        "user and device attempting access, regardless of their network location. "
        "Key implementation strategies include micro-segmentation and strong identity governance (IAM). "
        "This approach effectively counters insider threats and protects dispersed assets."
    )
    SAMPLE_IMAGES = [
        "IMAGE_ZTA_01: Network diagram with micro-segments and firewalls", 
        "IMAGE_ZTA_02: Lock icon with 'Verify' label over a login screen"
    ]
    
    print("Loading fine-tuned Phi model and adapters...")
    try:
        model, tokenizer = load_fine_tuned_model()
    except Exception as e:
        print(f"FATAL ERROR during model loading: {e}. Check ADAPTER_PATH and VRAM.")
        exit()

    print("\nGenerating structured JSON output...")
    
    # --- Execute the generation ---
    ppt_structure = generate_ppt_json(SAMPLE_RAW_TEXT, SAMPLE_IMAGES, model, tokenizer)

    if ppt_structure:
        print("\n" + "="*50)
        print("✅ SUCCESS: Structured JSON Generated")
        print("="*50)
        print(json.dumps(ppt_structure, indent=2))
        print("\nReady for PPTX Assembly (Step 3).")
    else:
        print("\n❌ FAILED to generate valid JSON structure.")
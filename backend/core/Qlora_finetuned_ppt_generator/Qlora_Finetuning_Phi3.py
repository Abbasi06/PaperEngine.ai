import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

MODEL_NAME = "microsoft/phi-2"
DATASET_PATH = "training_data.jsonl"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    quantization_config=bnb_config, 
    device_map="auto"
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=32, 
    lora_alpha=64, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
print("Model prepared for QLoRA fine-tuning.")
model.print_trainable_parameters()

training_arguments = TrainingArguments(
    output_dir="./phi_json_model",
    num_train_epochs=5,                  # Iterations over the small dataset
    per_device_train_batch_size=2,       # Start with 4, adjust based on VRAM
    gradient_accumulation_steps=8,       # Simulates a larger batch size (4 * 8 = 32)
    learning_rate=2e-5,                  # Optimal for QLoRA
    logging_steps=2,
    save_strategy="epoch",
    optim="paged_adamw_8bit",            # Critical for memory management
    fp16=False,                          # Use bfloat16 if possible
    bf16=True                       # Recommended for modern GPUs (A100, H100, RTX 30/40 series)

)


dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

def formatting_prompts_func(examples):
    """Formats the data into the structure the model will be trained on."""
    output_texts = []
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        # The instruction format that the model learns: INSTRUCTION + INPUT -> OUTPUT
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Output:\n{output}{tokenizer.eos_token}"
        output_texts.append(text)
    return output_texts


# Your function to format the text remains the same
def formatting_prompts_func(examples):
    """Formats the data into the structure the model will be trained on."""
    output_texts = []
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        # The instruction format that the model learns: INSTRUCTION + INPUT -> OUTPUT
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Output:\n{output}{tokenizer.eos_token}"
        output_texts.append(text)
    return output_texts

# --- NEW: Tokenization Function ---
def tokenize_function(examples):
    texts = formatting_prompts_func(examples)
    
    tokenized_output = tokenizer(
        texts,
        max_length=1024,
        truncation=True,
        padding="max_length",
    )
    
    # CRITICAL: Create the 'labels' column
    tokenized_output["labels"] = tokenized_output["input_ids"].copy()
    
    return tokenized_output

# Apply the tokenization to the dataset
tokenized_dataset = dataset.map(
    tokenize_function, 
    batched=True, 
    # --> THIS LINE IS CRITICAL: remove the original columns
    remove_columns=['instruction', 'input', 'output'], 
)
# -------------------------------------------------------------

# 7. Trainer Initialization and Start (Check variable)
trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=tokenized_dataset, # <-- Must use 'tokenized_dataset'
    # ... (rest of Trainer initialization)
)

trainer.train() # UNCOMMENT TO START TRAINING
trainer.model.save_pretrained("./final_phi_json_adapter")
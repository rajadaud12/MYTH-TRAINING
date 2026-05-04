from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# =========================
# Config
# =========================
max_seq_length = 2048
dtype = None
load_in_4bit = True

# =========================
# Load Model
# =========================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# =========================
# LoRA Setup (FIXED)
# =========================
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# =========================
# Dataset Load
# =========================
dataset = load_dataset(
    "json",
    data_files="myth-dataset.jsonl",
    split="train"
)

# =========================
# FORMAT YOUR DATA (IMPORTANT FIX)
# =========================
def format_example(example):
    return {
        "text": f"""### Instruction:
{example['input']}

### Response:
{example['output']}"""
    }

dataset = dataset.map(format_example)

# Fix tokenizer padding
tokenizer.pad_token = tokenizer.eos_token

# =========================
# Trainer
# =========================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=1000,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        output_dir="outputs",
        optim="adamw_8bit",
        report_to="none",
    ),
)

# =========================
# Train
# =========================
trainer.train()

# =========================
# Merge LoRA before export
# =========================
model = model.merge_and_unload()

# =========================
# Export GGUF for Ollama
# =========================
model.save_pretrained_gguf(
    "model_gguf",
    tokenizer,
    quantization_method="q4_k_m"
)
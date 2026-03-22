import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig

# =====================
# MODEL CONFIG
# =====================
model_name = "microsoft/Phi-3.5-mini-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("🚀 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
    attn_implementation="eager"
)

model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# =====================
# LoRA CONFIG
# =====================
peft_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear"
)


# =====================
# LOAD DATA FROM JSON
# =====================
print("📂 Loading dataset from nyaya_clean_data.json...")

dataset = load_dataset(
    "json",
    data_files="nyaya_clean_data.json"
)["train"]

print(f"✅ Loaded {len(dataset)} training samples")

# =====================
# TRAINING CONFIG
# =====================
training_args = SFTConfig(
    output_dir="./results",
    dataset_text_field="text",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,      # 🔥 aur kam LR
    logging_steps=10,
    max_steps=800,           # 🔥 zyada steps, smooth learning
    fp16=False,
    bf16=False,
    save_strategy="no",
)

# =====================
# TRAINER
# =====================
print("🔥 Starting training...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
)

trainer.train()
print("✅ Training complete!")

# =====================
# SAVE LORA
# =====================
trainer.model.save_pretrained("./nyaya_lora")
tokenizer.save_pretrained("./nyaya_lora")

print("✅ LoRA adapter saved!")

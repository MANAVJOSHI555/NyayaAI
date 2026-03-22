import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
LORA_PATH = "./nyaya_lora"

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    attn_implementation="eager",
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    is_trainable=False
)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

model.config.use_cache = False

prompt = (
    "### Instruction:\n"
    "What is Section 302 IPC?\n\n"
    "### Response:\n"
)

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(device)

print("\nGenerating response...\n")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,          
        repetition_penalty=1.2,   
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("🌍 Loading NLLB Model...")

MODEL_NAME = "facebook/nllb-200-distilled-600M"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

model.eval()

print("✅ NLLB Ready!")

# -------------------------------
# Hindi → English
# -------------------------------
def hi_to_en(text):
    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"],
            max_new_tokens=128,
            do_sample=False
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# -------------------------------
# English → Hindi
# -------------------------------
def en_to_hi(text):
    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["hin_Deva"],
            max_new_tokens=128,
            do_sample=False
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
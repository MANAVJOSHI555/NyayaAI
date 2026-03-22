from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os
import json  
import re    

app = Flask(__name__)

print("Loading the 365-Section BNS Database...")
try:
    with open('bns_data.json', 'r', encoding='utf-8') as f:
        bns_db = json.load(f)
    print(f"Success! {len(bns_db)} BNS laws loaded into memory.")
except FileNotFoundError:
    print("Warning: bns_data.json not found. Nyaya will rely on memory.")
    bns_db = {}

print("Flask is waking up Nyaya... (Please wait...)")
base_model_name = "microsoft/Phi-3.5-mini-instruct"
adapter_path = "./nyaya_lora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=False,
)

if not os.path.exists(adapter_path):
    print(f"Error: Cannot find {adapter_path}")
else:
    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=False)
    print("Nyaya is Online!")

def search_bns(query):
    query = query.lower()
    
    match = re.search(r'(?:section|sec)\s*(\d+[a-z]?)', query)
    if not match:
        match = re.search(r'\b(\d{1,3}[a-z]?)\b', query)

    if match:
        sec_num = match.group(1)
        target_key = f"section {sec_num}"
        target_key_float = f"section {sec_num}.0"
        
        if target_key in bns_db:
            return f"Section {sec_num}: {bns_db[target_key]}"
        elif target_key_float in bns_db:
            return f"Section {sec_num}: {bns_db[target_key_float]}"
        
        for key, value in bns_db.items():
            if target_key in key:
                return f"{key.title()}: {value}"

    ignore_words = [
        "what", "is", "the", "for", "a", "of", "in", "to", "under", "how", "can", 
        "tell", "me", "about", "someone", "if", "who", "commits", "happens", "does",
        "punishment", "penalty", "increase", "decrease", "law", "rule", "section", "sec"
    ]
    
    search_words = [word for word in query.replace("?", "").replace(",", "").split() if word not in ignore_words]
    
    best_match = None
    highest_score = 0
    
    for key, value in bns_db.items():
        val_lower = value.lower()
        score = sum(1 for word in search_words if val_lower.find(word) != -1)
        
        if score > highest_score:
            highest_score = score
            best_match = f"{key.title()}: {value}" 
            
    if highest_score > 0:
        return best_match
        
    return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    
    greetings = ["hi", "hii", "hello", "hey", "greetings", "namaste"]
    if user_input.lower() in greetings:
        return jsonify({"response": "Hello! I am Nyaya, your Legal AI. Please ask me a specific legal question regarding the BNS."})

    retrieved_info = search_bns(user_input)

    if retrieved_info:
        system_instruction = (
            f"You are Nyaya, a strict expert in the Bharatiya Nyaya Sanhita (BNS). "
            f"You MUST answer using ONLY these FACTS: {retrieved_info}\n"
            f"Do not use outside knowledge. If the facts do not explicitly mention the details asked, "
            f"say 'The BNS does not specify this detail.' Do not hallucinate penalties."
        )
        print(f"FACT FOUND! Injecting: {retrieved_info[:60]}...") 
    else:
        system_instruction = (
            "You are Nyaya. The user just asked about something that is NOT in the Bharatiya Nyaya Sanhita (BNS) database. "
            "You MUST reply EXACTLY with this sentence: 'I am sorry, but I am specifically trained on the Bharatiya Nyaya Sanhita (BNS). This topic falls outside my legal jurisdiction.' "
            "Do NOT provide any legal advice. Do NOT guess the penalty."
        )
        print(f"NO FACTS FOUND. Triggering Guardrails.")

    combined_input = f"{system_instruction}\n\nUser Question: {user_input}"
    
    prompt = f"<|user|>\n{combined_input}<|end|>\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs, 
        max_new_tokens=256,
        do_sample=True,
        temperature=0.1,
        repetition_penalty=1.02, 
        eos_token_id=tokenizer.eos_token_id, 
        pad_token_id=tokenizer.eos_token_id,
    )

    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:] 
    
    clean_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    if clean_response.startswith(user_input):
        clean_response = clean_response.replace(user_input, "").strip()
    if clean_response.startswith("User Question:"):
        clean_response = clean_response.split(user_input)[-1].strip()

    cut_off_words = ["**Note**", "Note:", "Disclaimer:", "Here's how we could proceed"]
    for word in cut_off_words:
        if word in clean_response:
            clean_response = clean_response.split(word)[0].strip()

    return jsonify({"response": clean_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

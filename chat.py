import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os

base_model_name = "microsoft/Phi-3.5-mini-instruct"
adapter_path = "./nyaya_lora" 

print(f"Loading the Base Brain: {base_model_name}...")

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

print(f"Loading Your Adapter from: {adapter_path}...")

if not os.path.exists(adapter_path):
    print(f"ERROR: The folder '{adapter_path}' does not exist!")
    exit()

model = PeftModel.from_pretrained(base_model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=False) 

print("\n  Nyaya AI is Ready! (Type 'exit' to stop)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    prompt = f"<|user|>\n{user_input}<|end|>\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs, 
        max_new_tokens=1024,      
        do_sample=True,          
        temperature=0.1,
        eos_token_id=tokenizer.eos_token_id, 
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "<|assistant|>" in response:
         clean_response = response.split("<|assistant|>")[-1].strip()
    elif "Assistant:" in response:
         clean_response = response.split("Assistant:")[-1].strip()
    else:
        clean_response = response.replace(prompt.replace("<|user|>\n", "").replace("<|end|>\n<|assistant|>\n", ""), "")

    print(f"Nyaya: {clean_response}")
    print("-" * 50)

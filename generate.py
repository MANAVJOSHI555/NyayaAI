import json
import requests
import os

# 1. The Raw URLs of the dataset files
urls = [
    "https://huggingface.co/datasets/Techmaestro369/indian-legal-texts-finetuning/resolve/main/ipc_qa.json",
    "https://huggingface.co/datasets/Techmaestro369/indian-legal-texts-finetuning/resolve/main/crpc_qa.json",
    "https://huggingface.co/datasets/Techmaestro369/indian-legal-texts-finetuning/resolve/main/constitution_qa.json"
]

formatted_data = []
total_count = 0

print("🚀 Starting Manual Download & Cleaning...")

for url in urls:
    filename = url.split("/")[-1]
    print(f"   Downloading {filename}...", end=" ")
    
    try:
        # Download the raw file
        response = requests.get(url)
        data = response.json()
        print(f"✅ Got {len(data)} rows.")
        
        # Clean and Format
        for row in data:
            # We ONLY look for 'question' and 'answer'. We ignore 'id' or anything else.
            if "question" in row and "answer" in row:
                q = row["question"].strip()
                a = row["answer"].strip()
                
                # Apply the Phi-3.5 Chat Template (Critical for smart answers)
                text = f"<|user|>\n{q}<|end|>\n<|assistant|>\n{a}<|end|>"
                
                formatted_data.append({"text": text})
                total_count += 1
                
    except Exception as e:
        print(f"❌ Failed: {e}")

# 2. Save the clean file locally
output_file = "nyaya_clean_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, indent=2)

print("="*50)
print(f"🎉 SUCCESS! Saved {total_count} clean examples to '{output_file}'.")
print("👉 You can now run train.py using this file.")
print("="*50)
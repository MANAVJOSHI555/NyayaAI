import json
import requests
import os

urls = [
    "https://huggingface.co/datasets/Techmaestro369/indian-legal-texts-finetuning/resolve/main/ipc_qa.json",
    "https://huggingface.co/datasets/Techmaestro369/indian-legal-texts-finetuning/resolve/main/crpc_qa.json",
    "https://huggingface.co/datasets/Techmaestro369/indian-legal-texts-finetuning/resolve/main/constitution_qa.json"
]

formatted_data = []
total_count = 0

print("Starting Manual Download & Cleaning...")

for url in urls:
    filename = url.split("/")[-1]
    print(f" Downloading {filename}...", end=" ")
    
    try:
        # Download the raw file
        response = requests.get(url)
        data = response.json()
        print(f"Got {len(data)} rows.")
        
        for row in data:
            if "question" in row and "answer" in row:
                q = row["question"].strip()
                a = row["answer"].strip()
                
                text = f"<|user|>\n{q}<|end|>\n<|assistant|>\n{a}<|end|>"
                
                formatted_data.append({"text": text})
                total_count += 1
                
    except Exception as e:
        print(f"Failed: {e}")

output_file = "nyaya_clean_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, indent=2)

print("="*50)
print(f"Saved {total_count} clean examples to '{output_file}'.")
print("="*50)

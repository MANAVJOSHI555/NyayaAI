# Nyaya AI ⚖️

*A privacy-first, locally deployed AI legal assistant trained specifically on the Bharatiya Nyaya Sanhita (BNS).*

## 📌 Problem
General-purpose Large Language Models (LLMs) are unreliable for legal advice due to high hallucination rates—they often confidently make up fake laws. Furthermore, uploading sensitive or confidential legal queries to proprietary cloud APIs poses a massive data privacy risk. There is a critical need for a localized, domain-specific AI that guarantees factual accuracy regarding Indian Law without compromising user data.

## 🚀 Approach
We built a Local RAG (Retrieval-Augmented Generation) system integrated with a custom fine-tuned Small Language Model (SLM). 

* **Base Model:** Microsoft `Phi-3.5-mini-instruct`, chosen for its excellent reasoning capabilities relative to its compact size.
* **Fine-Tuning:** Applied Parameter-Efficient Fine-Tuning (PEFT) using LoRA via the `TRL` (Transformer Reinforcement Learning) library on a custom, cleaned dataset (`nyaya_clean_data.json`).
* **Grounding:** Implemented a RAG architecture to fetch exact clauses from the BNS database before generation. This acts as a strict guardrail against hallucination.
* **Deployment:** The entire pipeline is designed to run strictly on local hardware. No legal queries are sent to external servers.

## 🔄 Iterations
1. **Data Curation:** Scraping, formatting, and structuring the BNS legal text into a clean JSON format for training.
2. **Base Model Optimization:** Integrating `BitsAndBytes` for 4-bit quantization (NF4) to ensure the model loads and runs efficiently on consumer-grade local GPUs without Out-Of-Memory (OOM) errors.
3. **Supervised Fine-Tuning (SFT):** Injecting domain-specific legal phrasing into the model using LoRA adapters (`r=4`, `alpha=8`) to adjust behavior and tone without causing catastrophic forgetting of its base knowledge.
4. **RAG Integration & Guardrails:** Building the retrieval pipeline to ensure the model only answers based on retrieved context. If the law isn't in the retrieved context, the model is trained to state that it doesn't know.

## 🧠 Key Design Choices
* **RAG over Pure Fine-Tuning:** Fine-tuning teaches the model *how* to talk (adopting a formal, legal tone), but RAG gives it the actual *facts*. We chose RAG as the primary source of truth to absolutely minimize legal hallucinations.
* **4-bit Quantization:** Essential for democratizing the project. By quantizing the model, it runs efficiently on standard hardware rather than requiring massive enterprise server clusters.
* **LoRA (Low-Rank Adaptation):** Allowed us to train the model quickly and save the learned weights as a tiny, manageable adapter rather than saving and transferring a massive, entirely new model.

## ⏱️ Daily Time Commitment
* Consistently dedicating 2-3 hours daily to balance data cleaning, pipeline architecture, model training, and local deployment debugging.

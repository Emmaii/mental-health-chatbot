import os
import pickle
import faiss
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

# Paths
CSV_PATH       = os.path.join("data", "faqs.csv")
INDEX_PATH     = os.path.join("vector_store", "faiss_index.index")
QUESTIONS_PATH = os.path.join("vector_store", "questions.pkl")
ANSWERS_PATH   = os.path.join("vector_store", "faq_answers.pkl")

# Model for embeddings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModel.from_pretrained(MODEL_NAME)

def embed_text(text: str) -> np.ndarray:
    """Return a mean-pooled embedding for a single string."""
    encoded = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoded)
    vec = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
    # normalize to unit length
    return vec / np.linalg.norm(vec)

def build_faiss_index():
    df = pd.read_csv(CSV_PATH).dropna(subset=["question", "answer"])
    vectors, questions, answers = [], [], []

    for idx, row in df.iterrows():
        q, a = str(row["question"]).strip(), row["answer"]
        if not q:
            continue
        try:
            emb = embed_text(q)
        except Exception as e:
            print(f"[❌] Embed failed row {idx}: {e}")
            continue
        vectors.append(emb)
        questions.append(q)
        answers.append(a)

    if not vectors:
        raise RuntimeError("No embeddings generated. Check CSV.")

    dim = len(vectors[0])
    # inner-product index == cosine when vectors normalized
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(vectors, dtype="float32"))

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(QUESTIONS_PATH, "wb") as f:
        pickle.dump(questions, f)
    with open(ANSWERS_PATH, "wb") as f:
        pickle.dump(answers, f)

    print(f"[✅] Indexed {len(questions)} FAQs to {INDEX_PATH}")

if __name__ == "__main__":
    build_faiss_index()

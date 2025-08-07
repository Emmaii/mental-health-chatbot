import os
import pickle
import numpy as np
import faiss

from scripts.embed_and_index import embed_text
from scripts.chat_with_llm import query_llm

# Paths
INDEX_PATH     = os.path.join("vector_store", "faiss_index.index")
QUESTIONS_PATH = os.path.join("vector_store", "questions.pkl")
ANSWERS_PATH   = os.path.join("vector_store", "faq_answers.pkl")

# Load index & data
index = faiss.read_index(INDEX_PATH)
with open(QUESTIONS_PATH, "rb") as f:
    faq_questions = pickle.load(f)
with open(ANSWERS_PATH, "rb") as f:
    faq_answers = pickle.load(f)

# Cosine similarity threshold (0–1)
SIMILARITY_THRESHOLD = 0.2  

def retrieve_answer(user_input: str) -> str:
    # 1) Embed user query
    q_emb = embed_text(user_input).astype("float32").reshape(1, -1)
    # 2) FAISS search
    scores, indices = index.search(q_emb, k=1)
    best_score, best_idx = float(scores[0][0]), int(indices[0][0])

    print(f"[DEBUG] Best similarity: {best_score:.4f} — Q: {faq_questions[best_idx]}")

    # 3) If low confidence, fallback to LLM
    if best_score < SIMILARITY_THRESHOLD:
        print("[INFO] Falling back to LLM.")
        return query_llm(user_input)

    # 4) Otherwise, return FAQ answer
    return faq_answers[best_idx]

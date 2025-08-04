import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer

# === Load vector index and metadata ===
base_path = "../vector_store"
index = faiss.read_index(os.path.join(base_path, "faiss_index.index"))

with open(os.path.join(base_path, "questions.pkl"), "rb") as f:
    questions = pickle.load(f)

with open(os.path.join(base_path, "faq_answers.pkl"), "rb") as f:
    answers = pickle.load(f)

# === Load embedding model ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Define retrieval function ===
def retrieve_faq_answer(user_input, top_k=3):
    query_vector = model.encode([user_input])
    distances, indices = index.search(query_vector, top_k)

    results = []
    for idx in indices[0]:
        question = questions[idx]
        answer = answers[idx]
        results.append((question, answer))

    return results

# === Test the retrieval ===
if __name__ == "__main__":
    print("\n🤖 Mental Health FAQ Chatbot (Test Mode)\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        hits = retrieve_faq_answer(query)
        print("\nTop matches:")
        for q, a in hits:
            print(f"\nQ: {q}\nA: {a}")
        print("-" * 40)

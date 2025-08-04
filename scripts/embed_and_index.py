import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

# Step 1: Load FAQ CSV
csv_path = os.path.join("data", "faqs.csv")
df = pd.read_csv(csv_path)
questions = df["question"].tolist()
answers = df["answer"].tolist()

# Step 2: Load embedding model
print("🔍 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 3: Encode questions
print("⚙️ Generating embeddings...")
embeddings = model.encode(questions, convert_to_numpy=True)

# Step 4: Build FAISS index
print("📦 Indexing with FAISS...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Step 5: Save everything
output_dir = "../vector_store"
os.makedirs(output_dir, exist_ok=True)

faiss.write_index(index, os.path.join(output_dir, "faiss_index.index"))

with open(os.path.join(output_dir, "faq_answers.pkl"), "wb") as f:
    pickle.dump(answers, f)

with open(os.path.join(output_dir, "questions.pkl"), "wb") as f:
    pickle.dump(questions, f)

print("✅ Done! Vector index and data saved.")

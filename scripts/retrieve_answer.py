import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scripts.chat_with_llm import query_llm

# Load embedded questions and their vectors
embedded_data_path = "data/embedded_faqs.csv"

def load_embeddings():
    df = pd.read_csv(embedded_data_path)
    df['embedding'] = df['embedding'].apply(lambda x: np.fromstring(x.strip("[]"), sep=","))
    return df

def get_top_contexts(user_question, df, top_k=3):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, lightweight

    user_embedding = model.encode([user_question])
    faq_embeddings = np.stack(df['embedding'].values)

    similarities = cosine_similarity(user_embedding, faq_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    top_contexts = df.iloc[top_indices]['question'].tolist()
    return top_contexts

def generate_prompt(user_question, contexts):
    context_block = "\n".join([f"- {ctx}" for ctx in contexts])
    return f"""
You are a mental health support assistant. Use the information below to answer the user's question as helpfully and kindly as possible.

Context:
{context_block}

User: {user_question}
Answer:
""".strip()

def get_answer(user_question):
    df = load_embeddings()
    top_contexts = get_top_contexts(user_question, df)
    prompt = generate_prompt(user_question, top_contexts)
    return query_llm(prompt)

import os
from openai import OpenAI
from retrieve_answer import retrieve_faq_answer

# ====== Create OpenAI Client ======
client = OpenAI()  # Automatically uses OPENAI_API_KEY from env

# ====== Build prompt template ======
def build_prompt(contextual_faqs, user_query):
    prompt = "You are a mental health support assistant. You do not give medical advice.\n"
    prompt += "Based on the following FAQ answers, respond kindly and concisely to the user's question.\n\n"
    
    for i, (q, a) in enumerate(contextual_faqs, 1):
        prompt += f"FAQ {i}:\nQ: {q}\nA: {a}\n\n"
    
    prompt += f"User's Question: {user_query}\n"
    prompt += "Your Response:"
    return prompt

# ====== Send to LLM ======
def ask_llm(user_query):
    faqs = retrieve_faq_answer(user_query)
    prompt = build_prompt(faqs, user_query)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful mental health assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=250
    )

    return response.choices[0].message.content.strip()

# ====== CLI test loop ======
if __name__ == "__main__":
    print("\n💬 Mental Health LLM Chatbot (Test Mode)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        reply = ask_llm(user_input)
        print(f"Bot: {reply}")
        print("-" * 50)


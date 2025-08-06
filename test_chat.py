from scripts.chat_with_llm import query_llm

print("🤖 Mental Health Chatbot is online. Type 'exit' to quit.\n")

while True:
    prompt = input("You: ")
    if prompt.lower() in ["exit", "quit"]:
        print("Bot: Take care! 💙")
        break

    response = query_llm(prompt)
    print("Bot:", response)

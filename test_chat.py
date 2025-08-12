from scripts.retrieve_answer import retrieve_answer

def main():
    print("Mental Health Chatbot (CLI)")
    print("Type 'exit' to quit.\n")
    while True:
        user = input("You: ")
        if user.lower() in ("exit", "quit"):
            break
        print("Bot:", retrieve_answer(user))
    print("Goodbye!")

if __name__ == "__main__":
    main()

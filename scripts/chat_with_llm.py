import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY  = os.getenv("OPENROUTER_API_KEY")
MODEL_ID = os.getenv("MODEL_ID")  # e.g. mistralai/mistral-7b-instruct
API_URL  = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def query_llm(prompt: str) -> str:
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": "You are a compassionate mental health assistant."},
            {"role": "user",   "content": prompt}
        ]
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("❌ LLM error:", e)
        return "⚠️ Sorry, I’m having trouble right now."

import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_ID = os.getenv("MODEL_ID")

API_URL = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def query_llm(prompt):
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": "You are a compassionate mental health assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as http_err:
        print(f"❌ HTTP Error: {http_err}")
        print(f"💬 Server says: {response.text}")
        return "⚠️ Could not process your request."
    except Exception as e:
        print(f"❌ Error: {e}")
        return "⚠️ Something went wrong."


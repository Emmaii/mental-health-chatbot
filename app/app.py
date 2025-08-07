import streamlit as st
import openai
import os, sys

# Ensure the project root is on sys.path so we can import scripts/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.retrieve_answer import retrieve_answer
openai.api_key = st.secrets["OPENROUTER_API_KEY"]

openai.api_base = "https://openrouter.ai/api/v1"

st.set_page_config(page_title="Mental Health Chatbot")
st.title("🧠 Mental Health Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask me anything about mental health…", "")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    st.session_state.messages.append(("user", user_input))
    # show spinner while retrieving
    with st.spinner("🤖 Thinking…"):
        bot_reply = retrieve_answer(user_input)
    st.session_state.messages.append(("bot", bot_reply))


for role, msg in st.session_state.messages:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)


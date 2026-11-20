import os
from pathlib import Path

import streamlit as st
from groq import Groq

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Code Tutor AI",
    page_icon="💻",
    layout="wide"
)

st.title("💻 Code Tutor AI")
st.caption("Your AI programming assistant for Python, Java and C")

# ---------------- LOAD KNOWLEDGE BASE ----------------
KNOWLEDGE_FILE = Path(__file__).parent / "programming_knowledge_base.txt"

def load_knowledge():
    if KNOWLEDGE_FILE.exists():
        return KNOWLEDGE_FILE.read_text(encoding="utf-8")
    return ""

knowledge_base = load_knowledge()

if not knowledge_base:
    st.error(
        "Knowledge base file not found! "
        "Place programming_knowledge_base.txt "
        "in the same folder as app.py."
    )
    st.stop()

# ---------------- GROQ API KEY ----------------
try:
    api_key = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    api_key = ""

api_key = api_key or os.getenv("GROQ_API_KEY", "")

if not api_key:
    st.error("Groq API key is missing. Configure GROQ_API_KEY in Streamlit Secrets.")
    st.stop()

client = Groq(api_key=api_key)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    language = st.selectbox(
        "Choose Programming Language",
        ["Python", "Java", "C", "General Programming"]
    )

    model = st.selectbox(
        "Choose Groq Model",
        ["llama-3.3-70b-versatile"]
    )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.success("Knowledge Base Loaded")
    st.caption("Python + Java + C")

# ---------------- CHAT HISTORY ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- SYSTEM PROMPT ----------------
system_prompt = f"""
You are Code Tutor AI, an expert programming assistant.

Selected language: {language}

Use the following programming knowledge base as your reference:

{knowledge_base}

INSTRUCTIONS:
1. Answer questions clearly and accurately.
2. Prefer the selected programming language.
3. Explain code step by step.
4. Provide complete code examples when useful.
5. Use correct Markdown code blocks.
6. Help users debug programming errors.
7. Explain the cause of errors and provide corrected code.
8. If information is not available in the knowledge base,
   use your general programming knowledge.
9. Never invent execution results.
10. Keep answers beginner-friendly.
"""

# ---------------- USER INPUT ----------------
prompt = st.chat_input("Ask your programming question...")

if prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    api_messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    api_messages.extend(st.session_state.messages)

    with st.chat_message("assistant"):

        try:
            with st.spinner("🤖 AI is thinking..."):

                response = client.chat.completions.create(
                    model=model,
                    messages=api_messages,
                    temperature=0.3,
                    max_tokens=2500
                )

                answer = (
                    response.choices[0].message.content
                    or "Sorry, I could not generate an answer."
                )

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:
            st.error(f"Groq API Error: {e}")

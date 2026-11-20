import streamlit as st
from groq import Groq

# Page settings
st.set_page_config(
    page_title="Programming AI Assistant",
    page_icon="💻",
    layout="wide"
)

st.title("💻 Programming AI Assistant")
st.caption("Ask questions about Python, Java and C     created by harshiiii")

# Connect to Groq
try:
    api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=api_key)
except Exception:
    st.error("Groq API key is missing. Add it in Streamlit Secrets.")
    st.stop()

# Current supported model
MODEL = "openai/gpt-oss-120b"

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful programming tutor. "
                "Teach Python, Java and C with clear explanations, "
                "correct code examples and step-by-step guidance."
            )
        }
    ]

# Display chat history
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User input
prompt = st.chat_input("Ask your programming question...")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=st.session_state.messages,
                    temperature=0.5,
                    max_completion_tokens=2048
                )

                answer = response.choices[0].message.content
                st.markdown(answer)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as error:
                st.error(f"Groq API Error: {error}")

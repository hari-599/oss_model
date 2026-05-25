import os

import streamlit as st
from dotenv import load_dotenv

from assistant import FoundationAssistant


load_dotenv()


st.set_page_config(
    page_title="Foundation Model Assistant",
    layout="centered",
)

st.title("Foundation Model Assistant")
st.caption("Multi-turn chat with memory, guardrails, eval hooks, and observability.")


with st.sidebar:
    st.header("Settings")

    model = st.text_input("Model", value=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    st.caption("Use an API model id such as gemini-2.5-flash.")
    memory_turns = st.slider("Memory turns", min_value=2, max_value=20, value=8)
    max_tokens = st.slider("Max output tokens", min_value=100, max_value=1500, value=500)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Observability events are written to logs/events.jsonl.")


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi, I am Luna. What would you like help with?",
        }
    ]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if user_input := st.chat_input("Ask Luna anything useful"):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    assistant = FoundationAssistant(
        model=model,
        memory_turns=memory_turns,
        max_tokens=max_tokens,
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = assistant.generate(st.session_state.messages)
        st.markdown(response.text)
        st.caption(
            f"{response.provider} | {response.model} | "
            f"{response.latency_ms} ms | trace {response.trace_id}"
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.text,
        }
    )

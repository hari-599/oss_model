import streamlit as st
from assistant import ChatAssistant
st.set_page_config(page_title="My Streamlit App", page_icon=":sparkles:", layout="centered")
st.title("Luna!")
st.write("Luna is a OSS model based chat assistant to help you!")

@st.cache_resource
def load_chat_assistant():
    return ChatAssistant()

assistant=load_chat_assistant()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        assistant.history = []
        st.rerun()

    st.markdown("---")
    st.markdown("### Model")
    st.write("Qwen2.5-1.5B-Instruct")

    st.markdown("### Features")
    st.write("✅ Multi-turn memory")
    st.write("✅ Conversational assistant")
    st.write("✅ Open-source model")


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
user_input = st.chat_input("Type your message...")


if user_input:
    # Store and display user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = assistant.generate_response(user_input)

        st.markdown(response)

    # Store assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

st.markdown("---")
st.subheader("Recent Chats")

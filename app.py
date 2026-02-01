import streamlit as st
from agent import run_agent, stream_chat_tokens, is_calculator_query
from langchain.schema import HumanMessage, AIMessage

st.set_page_config(
    page_title="LangGraph Agent with Groq",
    page_icon="🤖",
    layout="centered",
)

st.title("LangGraph Agent with Groq")
st.caption("Chat with streaming. Try «calculate 2 + 2» for the calculator.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history (user / assistant bubbles)
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Chat input at bottom (ChatGPT-style)
if prompt := st.chat_input("Type your message..."):
    # Append user message and show it
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant reply: stream tokens for chat, full reply for calculator
    with st.chat_message("assistant"):
        if is_calculator_query(prompt):
            # Use full graph (no streaming)
            updated = run_agent(prompt, st.session_state.messages[:-1])
            st.session_state.messages = updated
            st.markdown(st.session_state.messages[-1].content)
        else:
            # Stream LLM tokens
            messages = list(st.session_state.messages)
            full = st.write_stream(stream_chat_tokens(messages))
            st.session_state.messages.append(AIMessage(content=full))

st.markdown("---")
st.caption("Powered by Groq + LangGraph")

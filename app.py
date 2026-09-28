import streamlit as st
from chatbot import get_answer

st.set_page_config(page_title="FAQ Chatbot", page_icon="💬", layout="centered")

st.title("💬 Customer Support FAQ Chatbot")
st.caption("Ask anything about account, billing, orders, shipping, and more.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider(
        "Match threshold",
        0.0, 0.5, 0.05, 0.01,
        help="Lower = more matches. Higher = stricter."
    )
    st.caption("If you get 'Sorry' too often → lower this.")
    st.caption("If you get wrong answers → raise this.")

    st.markdown("---")
    st.markdown("**Try these:**")
    for q in [
        "How do I reset my password?",
        "I forgot my password",
        "cancel subscription",
        "refund time",
        "app keeps crashing",
        "track my order",
        "change credit card",
    ]:
        st.markdown(f"- {q}")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
user_input = st.chat_input("Type your question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    answer, score, matched = get_answer(user_input, threshold=threshold)

    with st.chat_message("assistant"):
        st.write(answer)
        with st.expander("🔍 Match details"):
            st.write(f"**Confidence:** {score:.2f}")
            if matched:
                st.write(f"**Matched FAQ:** {matched}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
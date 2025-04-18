import streamlit as st
from rag_chatbot import get_answer_gemini

st.set_page_config(page_title="JEE Doubt Solver 🤖", page_icon="📘")
st.title("📘 JEE RAG Chatbot")

st.markdown("Ask me any JEE Physics/Chemistry/Maths question. I’ll answer using your study notes!")

query = st.text_input("❓ Enter your question:")

if query:
    with st.spinner("🔍 Searching your notes and thinking..."):
        answer = get_answer_gemini(query)
        st.success("✅ Here's the answer:")
        st.markdown(f"**{answer}**")

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


genai.configure(api_key=GOOGLE_API_KEY)


embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.load_local("jee_faiss_index", embedding, allow_dangerous_deserialization=True)


def get_context(query, k=3):
    docs = vectorstore.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in docs])

def get_answer_gemini(query):
    context = get_context(query)
    
    prompt = f"""
You are an expert JEE tutor. Use the provided context to answer the student's question.

Context:
{context}

Question:
{query}

Answer:"""

    model = genai.GenerativeModel(model_name = "models/gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    return response.text.strip()

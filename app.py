import streamlit as st
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss, os, pickle
from sentence_transformers import SentenceTransformer
import numpy as np

client = OpenAI(api_key="YOUR_OPENAI_API_KEY")
model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("📚 Hack-a-Cure 9 Document Chatbot")

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
    st.session_state.text_chunks = []

uploaded_files = st.file_uploader("Upload the 9 source files", accept_multiple_files=True)

if uploaded_files:
    text = ""
    for file in uploaded_files:
        text += file.read().decode("utf-8", errors="ignore") + "\n"

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_text(text)

    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    st.session_state.faiss_index = index
    st.session_state.text_chunks = chunks
    st.success("✅ Files processed successfully!")

query = st.text_input("Ask a question about the documents:")
if query and st.session_state.faiss_index is not None:
    query_emb = model.encode([query])
    D, I = st.session_state.faiss_index.search(np.array(query_emb), k=3)
    context = "\n\n".join([st.session_state.text_chunks[i] for i in I[0]])

    prompt = f"Answer based only on this context:\n{context}\n\nQuestion: {query}"
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    st.write(response.choices[0].message.content)

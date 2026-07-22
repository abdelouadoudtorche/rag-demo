"""
Nova Gear Customer Support Bot — a RAG (Retrieval-Augmented Generation) demo.

Pipeline:
1. Load document -> split into chunks
2. Convert chunks into embeddings (local, free model)
3. Store embeddings in Chroma (local vector database)
4. On user question -> embed question -> retrieve top matching chunks
5. Send question + chunks to Groq LLM -> get grounded answer
6. Display everything in a Streamlit chat interface

Note: this version calls the retriever and the LLM directly (no LangChain
"chain" abstraction) to avoid breaking on LangChain's frequent internal
restructuring. Same RAG logic, fewer moving parts.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# ---- Load API key from .env file (never hardcode it) ----
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
st.set_page_config(page_title="Nova Gear Support Bot", page_icon="🎒")
st.title("🎒 Nova Gear — AI Customer Support")
st.caption("Ask about shipping, returns, products, or policies. Answers are grounded in Nova Gear's official FAQ.")
uploaded_file = st.file_uploader("Upload a document (.txt or .pdf)", type=["txt", "pdf"])
# ---- STEP 1 & 2 & 3: Build the knowledge base (only runs once, then cached) ----
@st.cache_resource
@st.cache_resource
def build_vectorstore(file_content, file_name):
    with open(file_name, "wb" if file_name.endswith(".pdf") else "w",
              encoding=None if file_name.endswith(".pdf") else "utf-8",
              errors=None if file_name.endswith(".pdf") else "ignore") as f:
        f.write(file_content)
    if file_name.endswith(".pdf"):
        loader = PyPDFLoader(file_name)
        documents = loader.load()
    else:
        with open(file_name, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        documents = [Document(page_content=text)]
    # ... rest stays exactly the same (splitter, embeddings, Chroma)
    # Split the document into small overlapping chunks.
    # Why chunks? LLMs and retrieval work better on small, focused pieces
    # rather than one giant blob of text.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,      # roughly 400 characters per chunk
        chunk_overlap=50,    # slight overlap so context isn't cut mid-thought
    )
    chunks = splitter.split_documents(documents)

    # STEP 2: Convert each chunk into an embedding (a vector of numbers).
    # This model runs locally on your machine — free, no API key needed.
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # STEP 3: Store all chunk-vectors in Chroma, a local vector database.
    # This lets us later search "which chunks are most similar to this question?"
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore

if uploaded_file is not None:
    if uploaded_file.name.endswith(".pdf"):
        file_content = uploaded_file.read()
    else:
        file_content = uploaded_file.read().decode("utf-8")

    vectorstore = build_vectorstore(file_content, uploaded_file.name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # ---- LLM setup: the model that generates the final answer ----
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0,
    )

    def answer_question(question: str):
        """Manual RAG: retrieve relevant chunks, build a prompt, call the LLM."""
        retrieved_docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        prompt = f"""You are a helpful customer support assistant.
Answer the customer's question using ONLY the context below. If the answer isn't in the context, say that
you don't have that information.

Context:
{context}

Question: {question}

Answer:"""

        response = llm.invoke(prompt)
        return response.content, retrieved_docs

    # ---- STEP 6: Streamlit chat interface ----
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_question = st.chat_input("Ask a question about the uploaded document...")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = answer_question(user_question)
                st.write(answer)

                with st.expander("Sources used"):
                    for doc in sources:
                        st.write(doc.page_content)
                        st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Please upload a document to get started.")
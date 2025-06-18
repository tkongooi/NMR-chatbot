import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# --- 1. Setup & API Key Loading ---
load_dotenv()
# Note: GOOGLE_API_KEY is loaded automatically by the ChatGoogleGenerativeAI class

# --- 2. Build the RAG components (using local data) ---
# This assumes you have already created a ChromaDB vector store named "my_documents"
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory="./my_documents", embedding_function=embedding_function)
retriever = vector_store.as_retriever()

# --- 3. Initialize the Remote LLM ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash") # Use a fast model for interactivity

# --- 4. Create the RAG Prompt and Chain ---
system_prompt = (
    "You are an expert assistant. Answer the user's question based only on the following context."
    "If the answer is not in the context, say so.\n\n"
    "<context>{context}</context>"
)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

Youtube_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, Youtube_chain)

# --- 5. Build the Streamlit Interface ---
st.title("My Personal Research Chatbot")
question = st.text_input("Ask a question about your documents:")

if question:
    with st.spinner("Searching documents and asking Gemini..."):
        response = rag_chain.invoke({"input": question})
        st.write(response["answer"])

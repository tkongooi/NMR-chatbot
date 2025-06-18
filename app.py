import streamlit as st
import os
from dotenv import load_dotenv

# Make sure to install these if you haven't:
# pip install streamlit langchain langchain-google-genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# --- 1. Load API Key and Basic Setup ---
# Load environment variables from .env file
load_dotenv()

# Check if the Google API key is available
if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found. Please create a .env file with your key.")
    st.stop()

# Define the paths for the vector store and source documents
PERSIST_DIRECTORY = "my_documents"
embedding_function = SentenceTransformerEmbeddings(model_name="thenlper/gte-large")

# --- 2. Load the Vector Store ---
# Check if the vector store directory exists
if not os.path.exists(PERSIST_DIRECTORY):
    st.error(f"Vector store not found at '{PERSIST_DIRECTORY}'.")
    st.error("Please run your `ingest.py` script first to create the knowledge base.")
    st.stop()

# Load the existing vector store
try:
    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_function
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 10}) # Retrieve top 10 most relevant chunks
except Exception as e:
    st.error(f"Failed to load the vector store: {e}")
    st.stop()


# --- 3. Initialize the LLM and Create the RAG Chain ---
# Use Google Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash") # Using flash for speed and cost-effectiveness

# Create a prompt template
system_prompt_old = (
    "You are an expert assistant for a university professor. Your expertise spans Dynamic Nuclear Polarization, "
    "Nuclear Magnetic Resonance, and finance. Answer the user's question based *only* on the provided context. "
    "If the answer is not in the context, state that clearly. Be concise and precise in your answers, citing "
    "the source document if possible.\n\n"
    "<context>{context}</context>"
)

# NEW HYBRID PROMPT
system_prompt = (
    "You are an expert assistant for a university professor. Your expertise spans Dynamic Nuclear Polarization, and "
    "Nuclear Magnetic Resonance.\n\n"
    "When answering, first check the user's provided context. "
    "If the answer is available in the context, use that as your primary source and begin your answer with 'According to your documents(internal sources),...'.\n\n"
    "If the answer is NOT in the context, you may use your general knowledge to provide a helpful response. "
    "In this case, start your answer with 'Based on my general knowledge (external sources),...'.\n\n"
    "If you cannot answer either way, say so.\n\n"
    "Context from user's documents:\n<context>{context}</context>"
)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

# Create the chains
Youtube_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, Youtube_chain)


# --- 4. Build the Streamlit User Interface ---
st.title("🔬 Kong's Research Assistant")
st.write("Ask me anything about your documents!")

# Persistent chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_question := st.chat_input("What are you wondering about?"):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # --- THIS IS THE CORRECTED LOGIC ---

    # 1. Retrieve documents ONCE.
    with st.spinner("Searching for relevant documents..."):
        retrieved_docs = retriever.get_relevant_documents(user_question)

    # 2. Display the retrieved documents in the debug view.
    with st.expander("🔍 See what the retriever found"):
        if not retrieved_docs:
            st.warning("The retriever found NO relevant documents.")
        for i, doc in enumerate(retrieved_docs):
            st.info(f"**Chunk {i+1}** (from file: `{doc.metadata.get('source', 'Unknown')}`)")
            st.write(doc.page_content)

    # 3. Generate the answer USING the retrieved documents.
    with st.spinner("Thinking..."):
        # This now passes the context explicitly to the chain.
        response = rag_chain.invoke({"input": user_question, "context": retrieved_docs})
        answer = response["answer"]

    # 4. Display the final answer.
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

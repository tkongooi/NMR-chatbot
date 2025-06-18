import os
from langchain_community.document_loaders import UnstructuredPDFLoader
import glob # Add this import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# Define the path to your source documents and the persistent database
SOURCE_DIRECTORY = "source_documents"
PERSIST_DIRECTORY = "my_documents"

def main():
    """
    Main function to ingest documents into the ChromaDB vector store.
    """
    print("--- Starting document ingestion process ---")

    # 1. Load Documents using the hi-res "unstructured" strategy
    print(f"Loading documents from: {SOURCE_DIRECTORY}")
    paths = glob.glob(os.path.join(SOURCE_DIRECTORY, "*.pdf"))
    if not paths:
        print("No PDF documents found in the source directory. Exiting.")
        return

    all_docs = []
    print("Using UnstructuredPDFLoader with 'ocr_only' strategy. This may be slow.")
    for file_path in paths:
        try:
            # Use the "ocr_only" strategy for difficult PDFs.
            # You can also try "hi_res" for a potentially even better (but slower) result.
            loader = UnstructuredPDFLoader(file_path, mode="single", strategy="ocr_only")
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")

    documents = all_docs
    # In this loader, each document is a page.
    print(f"Successfully loaded {len(documents)} pages from {len(paths)} PDF files.")

    # 2. Split Documents into Chunks
    # This is crucial for both performance and retrieval accuracy.
    # chunk_size: The number of characters in each chunk.
    # chunk_overlap: The number of characters to overlap between chunks to maintain context.
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Split documents into {len(texts)} chunks.")

    # 3. Create Embeddings (smaller 'all-MiniLM-L6-v2')
    # This uses a local SentenceTransformer model to turn text chunks into vectors.
    print("Creating embeddings... (This may take a while for large libraries)")
    embedding_function = SentenceTransformerEmbeddings(
    model_name="thenlper/gte-large",
    encode_kwargs={'batch_size': 16}  # Process in smaller batches, default = 32 (Kong)
)

    # 4. Create and Persist the Vector Store
    # This is the core step where ChromaDB is created.
    # - `documents=texts`: The document chunks to be stored.
    # - `embedding=embedding_function`: The function to use for creating embeddings.
    # - `persist_directory=PERSIST_DIRECTORY`: The folder where the database will be saved.
    print(f"Creating and persisting the vector store in: {PERSIST_DIRECTORY}")
    db = Chroma.from_documents(
        documents=texts,
        embedding=embedding_function,
        persist_directory=PERSIST_DIRECTORY
    )

    print("--- Ingestion Complete! ---")
    print(f"Your vector store is ready in the '{PERSIST_DIRECTORY}' folder.")


if __name__ == "__main__":
    main()

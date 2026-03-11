from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag.document_loader import load_documents


def create_vector_db():

    # Load documents
    docs = load_documents()

    print("Creating embeddings using HuggingFace model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create vector database
    vector_db = FAISS.from_documents(
        docs,
        embeddings
    )

    # Save vector DB locally
    vector_db.save_local("vector_db")

    print("Vector database created successfully!")
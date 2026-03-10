from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from rag.document_loader import load_documents

def create_vector_db():

    # Load documents
    docs = load_documents()

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create FAISS vector database
    vector_db = FAISS.from_documents(
        docs,
        embeddings
    )

    # Save database locally
    vector_db.save_local("vector_db")

    print("Vector database created successfully!")
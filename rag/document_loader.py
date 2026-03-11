import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "data"

def load_documents():

    documents = []

    if not os.path.exists(DATA_PATH):
        print("Data folder not found!")
        return []

    for file in os.listdir(DATA_PATH):

        if file.endswith(".pdf"):

            file_path = os.path.join(DATA_PATH, file)

            try:
                print(f"Loading {file}...")

                loader = PyMuPDFLoader(file_path)
                docs = loader.load()

                documents.extend(docs)

            except Exception as e:
                print(f"Skipping {file} due to error: {e}")

    if not documents:
        print("No documents loaded.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(documents)

    print(f"Loaded {len(split_docs)} text chunks from documents")

    return split_docs
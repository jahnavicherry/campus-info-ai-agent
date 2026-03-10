from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

DATA_PATH = "data"

def load_documents():

    documents = []

    for file in os.listdir(DATA_PATH):

        if file.endswith(".pdf"):

            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            docs = loader.load()

            documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(documents)

    return split_docs
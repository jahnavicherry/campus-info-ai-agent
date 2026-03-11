from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from rag.retriever import get_retriever
from transformers import pipeline


def create_qa_chain():

    retriever = get_retriever()

    llm_pipeline = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_length=512
    )

    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return qa_chain
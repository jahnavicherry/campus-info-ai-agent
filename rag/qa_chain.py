from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from rag.retriever import get_retriever
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


def create_qa_chain():

    retriever = get_retriever()

    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    llm_pipeline = pipeline(
        "text-generation",   # ← change here
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256
    )

    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return qa_chain
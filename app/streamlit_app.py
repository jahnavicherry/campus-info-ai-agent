import streamlit as st
from rag.qa_chain import create_qa_chain

st.set_page_config(page_title="Campus AI Assistant")

st.title("🎓 Campus Information AI Assistant")

st.write("Ask any question about the campus.")

qa_chain = create_qa_chain()

query = st.text_input("Enter your question")

if query:

    with st.spinner("Searching campus data..."):

        result = qa_chain.run(query)

    st.write("### Answer")
    st.write(result)
import streamlit as st
from haystack_exp.rag_pipeline import ask_question #, rag_pipeline # rag_pipeline is not directly used here anymore

# Initialize the RAG pipeline (if not already handled in ask_question or its module)
# This might be redundant if rag_pipeline.py handles its own initialization upon import.
# Consider if rag_pipeline.run() is called within ask_question or if the pipeline object itself needs to be present.
# If ask_question is truly self-contained, this line might not be necessary.
# However, to be safe and ensure all components are ready:
# rag_pipeline.warm_up() # Or any specific initialization method if warm_up() is not the one.
                        # If rag_pipeline object is not directly used by ask_question after initial setup,
                        # this might be better placed inside rag_pipeline.py to run on import.


st.title("RAM AWA Interview Q&A")

st.sidebar.header("OpenAI API Key")
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")


st.markdown("""
This is a simple Q&A demo using a RAG (Retrieval Augmented Generation) system.
Ask questions about the RAM AWA interview.
""")

user_question = st.text_input("Ask your question:")

if user_question:
    if not openai_api_key:
        st.warning("Please enter your OpenAI API Key in the sidebar to continue.")
        st.stop()
    if not openai_api_key.startswith('sk-'):
        st.warning("Please enter a valid OpenAI API Key, starting with 'sk-'.")
        st.stop()

    with st.spinner("Finding an answer..."):
        answer = ask_question(user_question, api_key=openai_api_key)
        st.write("Answer:")
        st.write(answer)

st.sidebar.header("About")
st.sidebar.info(
    "This app uses a Haystack RAG pipeline with a local embedding model "
    "to answer questions based on the RAM AWA interview transcripts."
) 
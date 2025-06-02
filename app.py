import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
import streamlit as st
from haystack_exp.rag_pipeline import ask_question #, rag_pipeline # rag_pipeline is not directly used here anymore

# Configuration for conversation history
MAX_HISTORY_LENGTH = 10

# Initialize the RAG pipeline (if not already handled in ask_question or its module)
# This might be redundant if rag_pipeline.py handles its own initialization upon import.
# Consider if rag_pipeline.run() is called within ask_question or if the pipeline object itself needs to be present.
# If ask_question is truly self-contained, this line might not be necessary.
# However, to be safe and ensure all components are ready:
# rag_pipeline.warm_up() # Or any specific initialization method if warm_up() is not the one.
                        # If rag_pipeline object is not directly used by ask_question after initial setup,
                        # this might be better placed inside rag_pipeline.py to run on import.


st.title("AWA Current Magazine QnA")

st.sidebar.header("OpenAI API Key")
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

# Initialize chat history in session state if it doesn't exist
if 'history' not in st.session_state:
    st.session_state.history = []

st.markdown("""
This is a simple Q&A demo using a RAG (Retrieval Augmented Generation) system.
Ask questions about the contents of  AWA Current Magazine .
""")

# Display chat history (oldest first)
if st.session_state.history:
    st.markdown("### Conversation History")
    for i, (q, a) in enumerate(st.session_state.history):
        st.markdown(f"**Q{i+1}:** {q}")
        st.markdown(f"**A{i+1}:** {a}")
        st.markdown("---")

# Create a form for user input
with st.form(key="chat_input_form"):
    user_question_text = st.text_input("Ask your question:")
    submit_button = st.form_submit_button(label="Ask")

if submit_button and user_question_text:
    if not openai_api_key:
        st.warning("Please enter your OpenAI API Key in the sidebar to continue.")
        st.stop()
    if not openai_api_key.startswith('sk-'):
        st.warning("Please enter a valid OpenAI API Key, starting with 'sk-'.")
        st.stop()

    with st.spinner("Finding an answer..."):
        answer = ask_question(user_question_text, api_key=openai_api_key)
        # Display the current answer (optional, as it will appear in history)
        # st.write("Answer:") 
        # st.write(answer)

        # Add current Q&A to history
        st.session_state.history.append((user_question_text, answer))
        # Keep only the last N conversations
        if len(st.session_state.history) > MAX_HISTORY_LENGTH:
            st.session_state.history = st.session_state.history[-MAX_HISTORY_LENGTH:]
        
        # Rerun to update the history display and show cleared input field
        st.rerun()


st.sidebar.header("About")
st.sidebar.info(
    "This app uses a Haystack RAG pipeline with a local embedding model "
    "to answer questions based on the RAM AWA interview transcripts."
) 
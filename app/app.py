################################################################################################################

#                                           Author: Anass MAJJI                                                #
 
#                                       File Name: streamlit_app.py                                            #

#                                       Creation Date: May 06, 2024                                            #

#                                        Source Language: Python                                               #

#                    Repository:    https://github.com/amajji/LLM-RAG-Chatbot-With-LangChain                   #

#                                         --- Code Description ---                                             #

#               Deploy LLM RAG Chatbot with Langchain on a Streamlit web application using only CPU            #

################################################################################################################


################################################################################################################
#                                               Packages                                                       #
################################################################################################################
import streamlit as st
import os
from llm import load_llm, model_retriever, q_a_llm_model
from vector_db import create_vector_db
from utils import configure_logging, get_memory_usage


################################################################################################################
#                                                Variables                                                     #
################################################################################################################

# Define Streamlit layout and global variables
st.set_page_config(layout="wide")
static_path = os.getenv("STREAMLIT_STATIC_PATH")
    

# Initialize logging
log_config_path = "/app/app/log.ini"
configure_logging(log_config_path)

################################################################################################################
#                                                main code                                                    #
################################################################################################################


# First streamlit's page
def page_1():

    # define the title
    st.title("✨ LLM with RAG")

    # quick decription of the webapp
    st.markdown(
        """
        This interactive dashboard allows users to extract information from external documents seamlessly. 
        Powered by the Llama 2-7B model and LangChain for retrieval-augmented generation (RAG), the app enables users to ask questions, 
        and the LLM delivers relevant answers based on the available documents.

        To enhance performance, we've optimized the model using the GGML quantization technique, reducing inference 
        time while maintaining accuracy. Notably, the application is designed to work efficiently even on CPU processors.
        """
    )

    st.markdown(
        """
        Below, a chat to interact with the LLM.
        """
    )

    # Text generation params
    st.sidebar.subheader("Text generation parameters")
    temperature = st.sidebar.slider(
        "Temperature", min_value=0.1, max_value=1.0, value=0.1, step=0.01
    )
    top_p = st.sidebar.slider(
        "Top_p", min_value=0.01, max_value=1.0, value=0.9, step=0.01
    )
    top_k = st.sidebar.slider("Top_k", min_value=0, max_value=100, value=20, step=10)
    max_length = st.sidebar.slider(
        "Max_length", min_value=64, max_value=4096, value=512, step=8
    )

    # Lazy load the vector DB and LLM only when needed
    if "vector_db" not in st.session_state:
        # Lazy load the vector database (FAISS index) when needed
        st.session_state["vector_db"] = create_vector_db(static_path)

    if "llm_model" not in st.session_state:
        # Lazy load the language model when needed
        st.session_state["llm_model"] = load_llm(temperature, max_length, top_p, top_k)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.caption("🚀 A Streamlit chatbot powered by OpenAI")
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Get the vector database and LLM model from session state
        vector_db = st.session_state["vector_db"]
        llm_model = st.session_state["llm_model"]

        # Model retriever should ideally be done only once and stored in session_state if not done already
        if "retriever" not in st.session_state:
            st.session_state["retriever"] = model_retriever(vector_db)

        # Use the retriever that is already created and stored
        retriever = st.session_state["retriever"]

        # Create the Q&A model (you could cache this too, or ensure it's created only once)
        if "q_a" not in st.session_state:
            st.session_state["q_a"] = q_a_llm_model(retriever, llm_model)

        # Use the stored Q&A model for the query
        q_a = st.session_state["q_a"]
        
        # Get the result from the Q&A model
        result = q_a.run({"query": prompt})

        st.session_state.messages.append(
            {"role": "assistant", "content": "voici le message de retour"}
        )
        st.chat_message("assistant").write(result)



def main():

    """A streamlit app template"""
    st.sidebar.title("Menu")

    PAGES = {
        "🎈 LLaMA2-7B LLM": page_1,
    }

    # Select pages
    # Use dropdown if you prefer
    selection = st.sidebar.radio("Select your page : ", list(PAGES.keys()))
    #st.sidebar_caption()

    PAGES[selection]()

    st.sidebar.title("About")
    st.sidebar.info(
        """
    Web App URL: <https://amajji-streamlit-dash-streamlit-app-8i3jn9.streamlit.app/>
    GitHub repository: <https://github.com/amajji/LLM-RAG-Chatbot-With-LangChain>
    """
    )

    st.sidebar.title("Contact")
    st.sidebar.info(
        """
    MAJJI Anass 
    [GitHub](https://github.com/amajji) | [LinkedIn](https://fr.linkedin.com/in/anass-majji-729773157)
    """
    )


if __name__ == "__main__":
    main()

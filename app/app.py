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
from dotenv import load_dotenv
from llm import load_llm, model_retriever, q_a_llm_model
from vector_db import create_vector_db, create_vector_db_from_uploaded_files
from utils import configure_logging, get_memory_usage

# Load environment variables from .env file
load_dotenv()


################################################################################################################
#                                                Variables                                                     #
################################################################################################################

# Define Streamlit layout and global variables
st.set_page_config(layout="wide")
# Set the path to the dataset directory containing PDF files
static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "pdf")
    

# Initialize logging
log_config_path = "/app/app/log.ini"
configure_logging(log_config_path)

################################################################################################################
#                                                main code                                                    #
################################################################################################################


# First streamlit's page
def page_1():
    # define the title
    st.title("✨ OpenAI RAG Chatbot")

    # quick description of the webapp
    st.markdown(
        """
        This interactive dashboard allows users to extract information from documents seamlessly. 
        Powered by OpenAI's GPT models and LangChain for retrieval-augmented generation (RAG), the app enables users to ask questions, 
        and the LLM delivers relevant answers based on your documents.

        **Two modes available:**
        - **Use existing vector database**: Connect to your pre-existing Pinecone vector database
        - **Upload new documents**: Upload PDF files to create a new knowledge base

        The application uses OpenAI's powerful language models (GPT-3.5-turbo, GPT-4) for generating responses, 
        combined with Pinecone vector database for efficient document retrieval and semantic search.
        """
    )

    # Get API keys from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    # Show status information
    if not api_key:
        st.error("⚠️ Please set your OpenAI API key in the .env file or as an environment variable OPENAI_API_KEY.")
        st.markdown("""
        **Setup Instructions:**
        1. Create a `.env` file in the project root
        2. Add: `OPENAI_API_KEY=your_api_key_here`
        3. Restart the application
        """)
        return
    
    if not pinecone_api_key:
        st.error("⚠️ Please set your Pinecone API key in the .env file or as an environment variable PINECONE_API_KEY.")
        st.markdown("""
        **Pinecone Setup Instructions:**
        1. Sign up at https://www.pinecone.io/
        2. Get your API key from the Pinecone console
        3. Add: `PINECONE_API_KEY=your_pinecone_api_key_here` to your .env file
        4. Optionally set: `PINECONE_ENVIRONMENT=us-east-1` and `PINECONE_INDEX_NAME=rag-chatbot`
        5. Restart the application
        """)
        return
    
    # Text generation params
    st.sidebar.subheader("Text generation parameters")
    temperature = st.sidebar.slider(
        "Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.01
    )
    max_tokens = st.sidebar.slider(
        "Max tokens", min_value=64, max_value=4096, value=512, step=8
    )
    model_name = st.sidebar.selectbox(
        "OpenAI Model", 
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
        index=0
    )

    # Document upload section
    st.subheader("📄 Document Management")
    
    # Add option to use existing vector database
    use_existing_db = st.checkbox(
        "Use existing vector database", 
        value=True,
        help="Check this to use the existing vector database from Pinecone without uploading new files."
    )
    
    if not use_existing_db:
        uploaded_files = st.file_uploader(
            "Choose PDF files to upload",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to create a knowledge base for the chatbot."
        )
    else:
        uploaded_files = None
        st.info("✅ Using existing vector database from Pinecone")

    # Process uploaded files and create vector database
    if uploaded_files and len(uploaded_files) > 0:
        if "uploaded_files_hash" not in st.session_state or st.session_state["uploaded_files_hash"] != str([f.name for f in uploaded_files]):
            with st.spinner("Processing uploaded documents..."):
                try:
                    # Create vector database from uploaded files
                    st.session_state["vector_db"] = create_vector_db_from_uploaded_files(uploaded_files)
                    st.session_state["uploaded_files_hash"] = str([f.name for f in uploaded_files])
                    st.session_state["uploaded_files_names"] = [f.name for f in uploaded_files]
                    
                    # Clear previous chat history when new documents are uploaded
                    st.session_state["messages"] = [
                        {"role": "assistant", "content": f"I've processed {len(uploaded_files)} document(s). How can I help you with questions about these documents?"}
                    ]
                    
                    # Clear cached models to force reload with new documents
                    if "retriever" in st.session_state:
                        del st.session_state["retriever"]
                    if "q_a" in st.session_state:
                        del st.session_state["q_a"]
                    
                    st.success(f"✅ Successfully processed {len(uploaded_files)} document(s)!")
                    
                except Exception as e:
                    st.error(f"Error processing documents: {e}")
                    return

    # Initialize vector database from existing Pinecone index if using existing DB
    if use_existing_db and "vector_db" not in st.session_state:
        with st.spinner("Connecting to existing vector database..."):
            try:
                from langchain_community.vectorstores import Pinecone
                from langchain_openai import OpenAIEmbeddings
                from pinecone import Pinecone as PineconeClient
                
                # Get configuration
                pinecone_api_key = os.getenv("PINECONE_API_KEY")
                openai_api_key = os.getenv("OPENAI_API_KEY")
                pinecone_index_name = os.getenv("PINECONE_INDEX_NAME") or "rag-chatbot"
                
                # Initialize Pinecone client
                pc = PineconeClient(api_key=pinecone_api_key)
                
                # Check if index exists
                existing_indexes = pc.list_indexes().names()
                if pinecone_index_name in existing_indexes:
                    # Initialize embeddings
                    embeddings = OpenAIEmbeddings(
                        model="text-embedding-3-small",
                        openai_api_key=openai_api_key
                    )
                    
                    # Connect to existing index
                    st.session_state["vector_db"] = Pinecone.from_existing_index(
                        index_name=pinecone_index_name,
                        embedding=embeddings
                    )
                    
                    # Initialize messages for existing database
                    if "messages" not in st.session_state:
                        st.session_state["messages"] = [
                            {"role": "assistant", "content": f"✅ Connected to existing vector database '{pinecone_index_name}'. How can I help you today?"}
                        ]
                    
                    st.success(f"✅ Connected to existing vector database: {pinecone_index_name}")
                else:
                    st.error(f"❌ Index '{pinecone_index_name}' not found in Pinecone. Please create an index first or upload documents.")
                    return
                    
            except Exception as e:
                st.error(f"Error connecting to existing vector database: {e}")
                return

    # Show uploaded files info
    if "uploaded_files_names" in st.session_state:
        st.info(f"📚 Loaded documents: {', '.join(st.session_state['uploaded_files_names'])}")

    # Initialize messages if not exists
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Please upload PDF documents or use existing vector database to get started!"}
        ]

    # Load LLM model
    if "llm_model" not in st.session_state:
        try:
            st.session_state["llm_model"] = load_llm(temperature, max_tokens, model_name)
        except Exception as e:
            st.error(f"Error loading OpenAI model: {e}")
            st.session_state["llm_model"] = None

    # Display chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Chat input - show if vector database is available (either uploaded or existing)
    if "vector_db" in st.session_state:
        prompt = st.chat_input("Ask a question about your uploaded documents...")
        
        if prompt:
            if "llm_model" not in st.session_state or st.session_state["llm_model"] is None:
                st.error("Please ensure your OpenAI API key is valid and try again.")
                return
                
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
            with st.spinner("Generating response..."):
                try:
                    result = q_a({"query": prompt})
                    answer = result["result"]
                    
                    # Show source documents if available
                    if "source_documents" in result and result["source_documents"]:
                        with st.expander("📖 Source Documents"):
                            for i, doc in enumerate(result["source_documents"]):
                                st.write(f"**Source {i+1}:**")
                                st.write(doc.page_content[:5000] + "..." if len(doc.page_content) > 5000 else doc.page_content)
                                st.write("---")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.chat_message("assistant").write(answer)
                    
                except Exception as e:
                    error_msg = f"Error generating response: {e}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.chat_message("assistant").write(error_msg)
    else:
        if use_existing_db:
            st.info("🔄 Connecting to existing vector database...")
        else:
            st.info("👆 Please upload PDF documents above to start chatting!")



def main():

    """A streamlit app template"""
    st.sidebar.title("Menu")

    PAGES = {
        "🤖 OpenAI RAG Chatbot": page_1,
    }

    # Select pages
    # Use dropdown if you prefer
    selection = st.sidebar.radio("Select your page : ", list(PAGES.keys()))
    #st.sidebar_caption()

    PAGES[selection]()

    st.sidebar.title("About")
    st.sidebar.info(
        """
    Aiinfox
    """
    )

    st.sidebar.title("Contact")
    st.sidebar.info(
        """
    Aiinfox
    """
    )


if __name__ == "__main__":
    main()

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
import logging
import time
import os



# Get the logger
logger = logging.getLogger()

def load_llm(temperature, max_tokens, model_name="gpt-3.5-turbo"):
    """Load the OpenAI LLM model"""
    try:
        logger.info("Start loading OpenAI LLM model ...")
        # start time
        start_time = time.time()

        # Get OpenAI API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        # Load the OpenAI model
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=api_key,
        )
        
        # end time
        end_time = time.time()
        logger.info(f"OpenAI model loaded successfully in {end_time - start_time: .2f} seconds")

        # return the LLM
        return llm
    except Exception as e:
        logger.error(f"Error loading OpenAI LLM model: {e}")
        raise



def model_retriever(vector_db):
    """
    This function creates a retriever object from the 'db' with a search configuration where
    it retrieves up to top_k relevant splits/documents.
    """
    try:
        retriever = vector_db.as_retriever(search_kwargs={"k": 2})   
        logger.info("Build retriever successfully.") 

        return retriever
    except Exception as e:
        logger.info(f"Error creating a retriever: {e}")
        raise

def q_a_llm_model(retriever, llm_model):
    """
    This function loads the LLM model, gets the relevant
    docs for a given query and provides an answer
    """
    try:
        logger.info(f"Start retrieving the relevant docs ...")

        # Create a question-answering instance (qa) using the RetrievalQA class.
        # Using "stuff" chain type for better document-based responses
        q_a = RetrievalQA.from_chain_type(
            llm=llm_model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
        return q_a
    except Exception as e:
        logger.error(f"Error creating Q&A LLM model: {e}")
        raise
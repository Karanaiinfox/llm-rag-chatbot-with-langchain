from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
import logging
import time



# Get the logger
logger = logging.getLogger()

def load_llm(temperature, max_new_tokens, top_p, top_k):
    """Load the LLM model"""
    
    logger.info("Start loading llm model with CTransformers ...")

    try:
        # start time
        start_time = time.time()

        # Load the locally downloaded model here
        llm = CTransformers(
            model="TheBloke/Llama-2-7B-Chat-GGML",
            model_type="llama",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        # end time
        end_time = time.time()
        logger.info(f"Model loaded with CTransformers successfully in {end_time - start_time: .2f} seconds")

        # List all available attributes for the model
        logger.info(llm.config)  

        # return the LLM
        return llm
    except Exception as e:
        logger.error(f"Error loading LLM model: {e}")
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
    This function loads the LLM model, gets the relevent
    docs for a given query and provides an answer
    """
    try:
        logger.info(f"Start retrieving the relevent docs ...")

        # Create a question-answering instance (qa) using the RetrievalQA class.
        q_a = RetrievalQA.from_chain_type(
            llm=llm_model,
            chain_type="refine",
            retriever=retriever,
            return_source_documents=False,
        )

        return q_a
    except Exception as e:
        logger.error(f"Error creating Q&A LLM model: {e}")
        raise
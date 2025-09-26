from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.prompts import PromptTemplate
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
        # Try similarity threshold first, fallback to regular similarity if not enough results
        try:
            retriever = vector_db.as_retriever(
                search_type="similarity_score_threshold",  # Use similarity threshold for better filtering
                search_kwargs={"k": 10, "score_threshold": 0.6}  # Lower threshold to get more results
            )
        except:
            # Fallback to regular similarity search if threshold search fails
            retriever = vector_db.as_retriever(
                search_kwargs={"k": 12}  # Get more chunks for comprehensive coverage
            )   
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

        # Custom prompt template to ensure PDF-only responses and comprehensive coverage
        prompt_template = """
        You are a helpful assistant that answers questions based ONLY on the provided documents. 
        
        CRITICAL INSTRUCTIONS:
        1. You MUST ONLY use information from the provided context documents below
        2. Do NOT use any external knowledge or general information
        3. If the answer is not in the provided documents, say "I cannot find this information in the uploaded documents"
        4. Provide COMPREHENSIVE and DETAILED answers covering ALL relevant points from the documents
        5. Include ALL sub-topics, details, and specific information mentioned in the documents
        6. If there are multiple aspects or sub-topics, cover ALL of them thoroughly
        7. Quote specific details, numbers, and examples from the documents when available
        8. Structure your response clearly with bullet points or numbered lists when appropriate
        
        Context documents:
        {context}
        
        Question: {question}
        
        Answer based ONLY on the context documents above. Be comprehensive and detailed:
        """

        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )

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
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient
import logging
import time
import tempfile
import os


# Get the logger
logger = logging.getLogger()


def create_vector_db(data_path):
    """function to create vector db provided the pdf files"""
    try:
        logger.info("Creation of the vector database ... ")
        
        # start time 
        start_time = time.time()

        # Get API keys from environment variables
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        pinecone_index_name = os.getenv("PINECONE_INDEX_NAME") or "rag-chatbot"
        
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set")
        
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        # Log configuration for debugging
        logger.info(f"Pinecone index name: {pinecone_index_name}")
        logger.info(f"Pinecone environment: {pinecone_environment}")

        # Initialize Pinecone client
        pc = PineconeClient(api_key=pinecone_api_key)

        # define the docs's path
        loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)

        # load documents
        documents = loader.load()

        # use recursive splitter to split each document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        texts = text_splitter.split_documents(documents)
        logger.info(" -- texts OK " )   

        # Initialize OpenAI embeddings model
        logger.info(" -- embeddings ... " )
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_api_key
        )
        
        # Validate index name
        if not pinecone_index_name or pinecone_index_name == "None":
            raise ValueError("PINECONE_INDEX_NAME environment variable is not set or is invalid")

        # Create or get Pinecone index
        existing_indexes = pc.list_indexes().names()
        logger.info(f"Existing indexes: {existing_indexes}")
        
        if pinecone_index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {pinecone_index_name}")
            pc.create_index(
                name=pinecone_index_name,
                dimension=1536,  # text-embedding-3-small produces 1536-dimensional embeddings
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": pinecone_environment}}
            )
        
        # Wait for index to be ready if it was just created
        if pinecone_index_name not in existing_indexes:
            logger.info("Waiting for index to be ready...")
            time.sleep(5)  # Wait 5 seconds for index to be ready
        
        # indexing database - use index_name parameter
        db = Pinecone.from_documents(
            texts, 
            embeddings, 
            index_name=pinecone_index_name
        )

        # end time
        end_time = time.time()

        logger.info(f"Vector DB created successfully in {end_time - start_time: .2f} seconds ")
        return db
    except Exception as e:
        logger.error(f"Error Creating vector database: {e}")
        raise


def create_vector_db_from_uploaded_files(uploaded_files):
    """function to create vector db from uploaded PDF files"""
    try:
        logger.info("Creation of the vector database from uploaded files... ")
        
        # start time 
        start_time = time.time()

        # Get API keys from environment variables
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        pinecone_index_name = os.getenv("PINECONE_INDEX_NAME") or "rag-chatbot"
        
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set")
        
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        # Log configuration for debugging
        logger.info(f"Pinecone index name: {pinecone_index_name}")
        logger.info(f"Pinecone environment: {pinecone_environment}")

        # Initialize Pinecone client
        pc = PineconeClient(api_key=pinecone_api_key)

        documents = []
        
        # Process each uploaded file
        for uploaded_file in uploaded_files:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Load the PDF document
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} pages from {uploaded_file.name}")
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)

        if not documents:
            raise ValueError("No documents were loaded from uploaded files")

        # use recursive splitter to split each document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(texts)} chunks")   

        # Initialize OpenAI embeddings model
        logger.info("Generating embeddings...")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_api_key
        )
        
        # Validate index name
        if not pinecone_index_name or pinecone_index_name == "None":
            raise ValueError("PINECONE_INDEX_NAME environment variable is not set or is invalid")

        # Create or get Pinecone index
        existing_indexes = pc.list_indexes().names()
        logger.info(f"Existing indexes: {existing_indexes}")
        
        if pinecone_index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {pinecone_index_name}")
            pc.create_index(
                name=pinecone_index_name,
                dimension=1536,  # text-embedding-3-small produces 1536-dimensional embeddings
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": pinecone_environment}}
            )
        
        # Wait for index to be ready if it was just created
        if pinecone_index_name not in existing_indexes:
            logger.info("Waiting for index to be ready...")
            time.sleep(5)  # Wait 5 seconds for index to be ready
        
        # indexing database - use index_name parameter
        db = Pinecone.from_documents(
            texts, 
            embeddings, 
            index_name=pinecone_index_name
        )

        # end time
        end_time = time.time()
        logger.info(f"Vector DB created successfully from uploaded files in {end_time - start_time: .2f} seconds")
        return db
    except Exception as e:
        logger.error(f"Error Creating vector database from uploaded files: {e}")
        raise
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
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

        # define the docs's path
        loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)

        # load documents
        documents = loader.load()

        # use recursive splitter to split each document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)

        texts = text_splitter.split_documents(documents)
        logger.info(" -- texts OK " )   

        # Initialize embeddings model with GPU support
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"
        
        # generate embeddings for each chunk
        logger.info(" -- embeddings ... " )
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            multi_process=True,
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={"device": device},
        )
        # indexing database
        db = FAISS.from_documents(texts, embeddings)

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

        # Initialize embeddings model
        device = "cpu"
        
        # generate embeddings for each chunk
        logger.info("Generating embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            multi_process=True,
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={"device": device},
        )
        
        # indexing database
        db = FAISS.from_documents(texts, embeddings)

        # end time
        end_time = time.time()
        logger.info(f"Vector DB created successfully from uploaded files in {end_time - start_time: .2f} seconds")
        return db
    except Exception as e:
        logger.error(f"Error Creating vector database from uploaded files: {e}")
        raise

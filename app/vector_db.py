from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import logging
import time


# Get the logger
logger = logging.getLogger()


def c(data_path):
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

# Environment Variables Configuration

## Required Environment Variables

### OpenAI Configuration
- `OPENAI_API_KEY`: Your OpenAI API key for GPT models and embeddings
  - Get it from: https://platform.openai.com/api-keys
  - Example: `OPENAI_API_KEY=sk-...`
  - Used for both: GPT models (chat completion) and text embeddings

### Pinecone Configuration
- `PINECONE_API_KEY`: Your Pinecone API key
  - Get it from: https://app.pinecone.io/organizations
  - Example: `PINECONE_API_KEY=your_pinecone_api_key_here`

### Optional Pinecone Configuration
- `PINECONE_ENVIRONMENT`: Pinecone environment/region (default: us-east-1)
  - Example: `PINECONE_ENVIRONMENT=us-east-1`
- `PINECONE_INDEX_NAME`: Name of the Pinecone index (default: rag-chatbot)
  - Example: `PINECONE_INDEX_NAME=rag-chatbot`

## Setup Instructions

1. Create a `.env` file in the project root directory
2. Add the required environment variables as shown above
3. Install dependencies: `pip install -r requirements.txt`
4. Run the application: `streamlit run app/app.py`

## Usage Modes

The application now supports two modes:

### Mode 1: Use Existing Vector Database (Default)
- ✅ **No file upload required**
- ✅ **Connect to existing Pinecone index**
- ✅ **Ready to chat immediately**
- Perfect for when you already have documents stored in Pinecone

### Mode 2: Upload New Documents
- 📄 **Upload PDF files to create new knowledge base**
- 🔄 **Process and store documents in Pinecone**
- 💬 **Start chatting with uploaded documents**
- Ideal for adding new documents to your knowledge base

## Migration from FAISS + HuggingFace to Pinecone + OpenAI

This project has been migrated from FAISS + HuggingFace to Pinecone + OpenAI:

### Benefits of Pinecone + OpenAI:
- **Scalability**: Handles large-scale vector operations
- **Performance**: Optimized for production workloads
- **Persistence**: Data persists across application restarts
- **Cloud-native**: Managed service with high availability
- **Advanced features**: Metadata filtering, hybrid search, etc.
- **Consistency**: Same API key for both LLM and embeddings
- **Quality**: OpenAI embeddings provide superior semantic understanding

### Changes Made:
1. Replaced `faiss-cpu` with `pinecone==5.4.2` in requirements.txt
2. Removed `sentence-transformers` and `transformers` dependencies
3. Updated `vector_db.py` to use OpenAI embeddings instead of HuggingFace
4. Added Pinecone configuration validation in `app.py`
5. Implemented automatic index creation with proper dimensions (1536 for text-embedding-3-small)
6. Fixed Pinecone package compatibility with Python 3.13
7. Updated Pinecone API usage to use `index_name` parameter instead of deprecated `pinecone_api_key`
8. Added proper index validation and creation timing to prevent "Index 'None' not found" errors
9. Made file upload optional - users can now use existing Pinecone vector database without uploading files
10. Added checkbox to toggle between existing database and new file upload modes

### Embedding Model Details:
- **Model**: `text-embedding-3-small`
- **Dimensions**: 1536
- **Provider**: OpenAI
- **Cost**: Very affordable ($0.00002 per 1K tokens)
- **Quality**: High-quality semantic embeddings

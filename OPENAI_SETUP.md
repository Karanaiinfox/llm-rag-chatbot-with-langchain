# OpenAI RAG Chatbot Setup

## Overview
This application has been converted from using local LLM (Llama-2-7B) to OpenAI's GPT models for better performance and ease of use.

## Setup Instructions

### 1. Install Dependencies
The required dependencies have been added to `requirements.txt`. Install them with:
```bash
pip install -r requirements.txt
```

### 2. Set up OpenAI API Key
You have two options to provide your OpenAI API key:

#### Option A: Environment Variable (Recommended)
Set the environment variable:
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your_api_key_here"

# Windows Command Prompt
set OPENAI_API_KEY=your_api_key_here

# Linux/Mac
export OPENAI_API_KEY="your_api_key_here"
```

#### Option B: Use the Web Interface
Enter your API key directly in the Streamlit sidebar when you run the application.

### 3. Run the Application
```bash
streamlit run ./app/app.py --server.port=8501 --server.address=0.0.0.0
```

## Features

### Available Models
- GPT-3.5-turbo (default, cost-effective)
- GPT-4 (higher quality, more expensive)
- GPT-4-turbo-preview (latest features)

### Configuration Options
- **Temperature**: Controls randomness (0.1 = deterministic, 1.0 = very random)
- **Max Tokens**: Maximum length of generated response
- **Model Selection**: Choose between different OpenAI models

### RAG Features
- Document ingestion from PDF files in `dataset/pdf/` directory
- Semantic search using FAISS vector database
- Context-aware responses based on your documents

## Usage
1. Start the application
2. Enter your OpenAI API key in the sidebar
3. Select your preferred model and parameters
4. Ask questions about your documents
5. The chatbot will provide answers based on the content in your PDF files

## Cost Considerations
- GPT-3.5-turbo: ~$0.0015 per 1K tokens
- GPT-4: ~$0.03 per 1K tokens
- Monitor your usage in the OpenAI dashboard

## Troubleshooting
- Ensure your OpenAI API key is valid and has sufficient credits
- Check that PDF files are present in the `dataset/pdf/` directory
- Verify all dependencies are installed correctly

#!/usr/bin/env python3
"""
Test script to verify Pinecone migration
This script tests the basic functionality of the migrated vector database
"""

import os
import sys
from dotenv import load_dotenv

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_imports():
    """Test that all required imports work"""
    try:
        from vector_db import create_vector_db, create_vector_db_from_uploaded_files
        from llm import load_llm, model_retriever, q_a_llm_model
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_environment_variables():
    """Test that required environment variables are set"""
    load_dotenv()
    
    required_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file")
        return False
    else:
        print("✅ All required environment variables are set")
        return True

def test_pinecone_client():
    """Test Pinecone client initialization"""
    try:
        from pinecone import Pinecone as PineconeClient
        load_dotenv()
        
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            print("❌ PINECONE_API_KEY not set")
            return False
            
        pc = PineconeClient(api_key=api_key)
        # Try to list indexes (this will fail if API key is invalid)
        indexes = pc.list_indexes()
        print("✅ Pinecone client initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Pinecone client error: {e}")
        return False

def test_openai_embeddings():
    """Test OpenAI embeddings initialization"""
    try:
        from langchain_openai import OpenAIEmbeddings
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not set")
            return False
            
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key
        )
        print("✅ OpenAI embeddings initialized successfully")
        return True
    except Exception as e:
        print(f"❌ OpenAI embeddings error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Pinecone + OpenAI Embeddings Migration...")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Environment Variables Test", test_environment_variables),
        ("Pinecone Client Test", test_pinecone_client),
        ("OpenAI Embeddings Test", test_openai_embeddings),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Migration appears successful.")
        print("\n📝 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set up your .env file with API keys")
        print("3. Run the app: streamlit run app/app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

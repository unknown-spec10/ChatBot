#!/usr/bin/env python3
"""
API Test Suite

Tests Groq API authentication and MiniLM embedding functionality.
Run this before using the main chatbot to verify your setup.

Usage:
    python tests/test_api.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_groq_api():
    """Test the Groq API connection with detailed debugging."""
    print("🔍 Testing Groq API Setup...")
    print("=" * 50)
    
    if 'GROQ_API_KEY' in os.environ:
        del os.environ['GROQ_API_KEY']
    
    env_loaded = load_dotenv(override=True)
    print(f"📁 .env file loaded: {env_loaded}")
    
    # Check if .env file exists
    if os.path.exists('.env'):
        print("✅ .env file found")
        with open('.env', 'r') as f:
            content = f.read()
            if 'GROQ_API_KEY=' in content:
                print("✅ GROQ_API_KEY found in .env file")
            else:
                print("❌ GROQ_API_KEY not found in .env file")
    else:
        print("❌ .env file not found")
        return False
    
    # Check if API key exists in environment
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ ERROR: GROQ_API_KEY not found in environment variables.")
        print("📝 Please ensure your .env file contains:")
        print("   GROQ_API_KEY=gsk_your_actual_api_key_here")
        return False
    
    # Show API key info
    print(f"🔑 API Key loaded: {api_key[:15]}...{api_key[-5:]}")
    print(f"📏 API Key length: {len(api_key)} characters")
    
    # Validate API key format
    if not api_key.startswith('gsk_'):
        print("❌ ERROR: API key should start with 'gsk_'")
        return False
    
    if len(api_key) < 40:
        print("❌ ERROR: API key appears to be too short")
        return False
    
    print("✅ API key format looks correct")
    
    # Test the API connection
    try:
        from groq import Groq
        
        print("🔧 Creating Groq client...")
        client = Groq(api_key=api_key)
        print("✅ Groq client initialized successfully")
        
        # Test API call with different approaches
        print("🧪 Testing API call...")
        
        # Try with the simplest possible request first
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": "Hello"}
            ],
            max_tokens=10
        )
        
        result = response.choices[0].message.content
        print(f"✅ API call successful!")
        print(f"📝 Response: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        error_str = str(e)
        
        if "401" in error_str:
            print("\n🔑 Authentication Error (401) - Possible causes:")
            print("   1. API key is incorrect or corrupted")
            print("   2. API key has been revoked or expired")
            print("   3. API key doesn't have proper permissions")
            print("   4. There might be a delay in key activation")
            print("\n💡 Try:")
            print("   - Wait 1-2 minutes and try again")
            print("   - Generate a completely new API key")
            print("   - Check your Groq console for key status")
            
        elif "404" in error_str:
            print("\n🤖 Model Error (404) - The model might not be available")
            print("   - Try a different model name")
            
        elif "429" in error_str:
            print("\n⏰ Rate Limit Error (429)")
            print("   - You've hit the rate limit")
            print("   - Wait a moment and try again")
            
        else:
            print(f"\n🔍 Unknown error type: {type(e)}")
            print("   - Check your internet connection")
            print("   - Verify Groq service status")
        
        return False

def test_minilm_embeddings():
    """Test the local MiniLM embedding service with detailed debugging."""
    print("\n🔍 Testing Local MiniLM Embeddings...")
    print("=" * 50)
    
    try:
        # Test import
        print("📦 Testing sentence-transformers import...")
        try:
            from sentence_transformers import SentenceTransformer
            print("✅ sentence-transformers imported successfully")
        except ImportError:
            print("❌ ERROR: sentence-transformers not installed")
            print("💡 Install it with: pip install sentence-transformers torch")
            return False
        
        # Test our custom embedding service
        print("🧪 Testing custom MiniLMEmbeddingService...")
        try:
            # Import our custom classes
            sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
            from src.config.settings import Configuration
            from src.rag.minilm_embeddings import MiniLMEmbeddingService
            
            print("✅ Successfully imported MiniLMEmbeddingService")
            
            # Initialize our service (this will download the model if not cached)
            print("🔧 Initializing MiniLM embedding service...")
            print("   (This may take a moment to download the model on first run)")
            
            config = Configuration()
            embedding_service = MiniLMEmbeddingService(config)
            
            print("✅ MiniLM embedding service initialized successfully")
            
            # Test connection
            print("🧪 Testing embedding service functionality...")
            if embedding_service.test_connection():
                print("✅ MiniLM embedding service connection test passed")
            else:
                print("❌ MiniLM embedding service connection test failed")
                return False
            
            # Test single embedding generation
            print("📝 Testing single text embedding...")
            test_text = "This is a test document for embedding generation."
            test_embedding = embedding_service.embed_text(test_text)
            
            print(f"✅ Single embedding successful!")
            print(f"📊 Embedding dimension: {len(test_embedding)}")
            print(f"📋 Expected dimension: {embedding_service.get_embedding_dimension()}")
            print(f"📝 First 5 values: {test_embedding[:5]}")
            
            # Validate embedding
            if len(test_embedding) != embedding_service.get_embedding_dimension():
                print("❌ ERROR: Embedding dimension mismatch")
                return False
            
            if all(val == 0 for val in test_embedding[:10]):
                print("❌ ERROR: Embedding contains only zeros")
                return False
            
            # Test query embedding
            print("🔍 Testing query embedding...")
            query_text = "What is this document about?"
            query_embedding = embedding_service.embed_query(query_text)
            
            print(f"✅ Query embedding successful!")
            print(f"📊 Query embedding dimension: {len(query_embedding)}")
            
            # Test batch embedding
            print("📦 Testing batch embedding...")
            batch_texts = [
                "First test document",
                "Second test document", 
                "Third test document"
            ]
            batch_embeddings = embedding_service.embed_batch(batch_texts)
            
            print(f"✅ Batch embedding successful!")
            print(f"📊 Generated {len(batch_embeddings)} embeddings")
            print(f"📋 All dimensions correct: {all(len(emb) == embedding_service.get_embedding_dimension() for emb in batch_embeddings)}")
            
            # Get model info
            model_info = embedding_service.get_model_info()
            print("📋 Model Information:")
            for key, value in model_info.items():
                print(f"   {key}: {value}")
            
            print("✅ All MiniLM embedding tests passed!")
            return True
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("💡 Make sure all dependencies are installed")
            return False
        except Exception as e:
            print(f"❌ MiniLM embedding service error: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Unexpected error in MiniLM test: {e}")
        return False

if __name__ == "__main__":
    print("Enhanced API Test Script")
    print("Testing Groq (chat) and MiniLM (local embeddings)")
    print("=" * 55)
    
    groq_success = test_groq_api()
    minilm_success = test_minilm_embeddings()
    
    print("\n" + "=" * 55)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 55)
    
    if groq_success:
        print("✅ Groq API (Chat): WORKING")
    else:
        print("❌ Groq API (Chat): FAILED")
    
    if minilm_success:
        print("✅ MiniLM Embeddings (Local): WORKING")
    else:
        print("❌ MiniLM Embeddings (Local): FAILED")
    
    if groq_success and minilm_success:
        print("\n🎉 SUCCESS! Both systems are working correctly.")
        print("💬 You can now run the chatbot with: python main.py")
        print("📚 You can also ingest documents with: python ingest_documents.py")
    else:
        print("\n❌ Some systems failed! Please fix the issues above.")
        
        if not groq_success:
            print("\n🔧 Groq API Issues:")
            print("   - Chat functionality will not work")
            print("   - Generate a Groq API key at: https://console.groq.com/")
            
        if not minilm_success:
            print("\n🔧 MiniLM Embedding Issues:")
            print("   - Document embeddings and RAG will not work")
            print("   - Install dependencies: pip install sentence-transformers torch")
        
        print("\n🆘 If problems persist:")
        print("   1. Ensure all required packages are installed")
        print("   2. Check that your .env file contains GROQ_API_KEY")
        print("   3. Verify you have sufficient disk space for models")
        print("   4. Check your internet connection for model downloads")
        
        sys.exit(1)
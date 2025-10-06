"""
Streamlit Cloud Deployment Entry Point

This is the main entry point that Streamlit Cloud expects for deployment.                            else:
                                vector_store = None
                        else:
                            vector_store = None
                            
                    except Exception:
                        vector_store = Noney handles secrets, environment configuration, and cloud-specific requirements.
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Clear any local environment GROQ_API_KEY to use Streamlit secrets
if 'GROQ_API_KEY' in os.environ:
    del os.environ['GROQ_API_KEY']

import streamlit as st
from src.config.settings import Configuration
from src.services.groq_provider import GroqAIProvider
from src.services.message_handler import ConversationHandler
from src.services.chatbot import AgenticChatbot
from src.exceptions.chatbot_exceptions import ConfigurationError, APIError
from src.rag.minilm_embeddings import MiniLMEmbeddingService
from src.rag.vector_store import EnhancedChromaVectorStore


def setup_cloud_environment():
    """Configure environment for Streamlit Cloud deployment."""
    # Use Streamlit secrets for API keys
    if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
        os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']
    
    # Set up logging for cloud environment
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]  # Only stream handler for cloud
    )


@st.cache_resource
def create_chatbot() -> AgenticChatbot:
    """
    Create and configure the chatbot for cloud deployment.
    
    Returns:
        Configured chatbot instance
        
    Raises:
        ConfigurationError: If configuration is invalid
        APIError: If AI provider setup fails
    """
    try:
        config = Configuration()
        ai_provider = GroqAIProvider(config)
        message_handler = ConversationHandler(
            max_history_length=config.get_max_history_length()
        )
        
        # Initialize RAG system for cloud
        try:
            embedding_service = MiniLMEmbeddingService(config)
            
            # Use a cloud-appropriate directory for vector store
            persist_dir = "./chroma_db"
            
            # Initialize vector store
            try:
                vector_store = EnhancedChromaVectorStore(
                    embedding_service=embedding_service,
                    persist_directory=persist_dir
                )
                
                # Check if pre-computed embeddings exist and auto-load if needed
                stats = vector_store.get_collection_stats()
                if stats.get('total_chunks', 0) == 0:
                    
                    # Try to auto-ingest documents silently using same strategy as ingestion script
                    try:
                        from src.rag.document_processor import DocumentProcessor
                        from pathlib import Path
                        
                        # Check for documents directory
                        data_dir = Path("./data")
                        documents_dir = data_dir / "documents"
                        
                        if documents_dir.exists() and any(documents_dir.iterdir()):
                            
                            processor = DocumentProcessor(
                                chunk_size=700,        # Target 700 tokens
                                chunk_overlap=100,     # Higher overlap for continuity
                                min_chunk_size=50      # Minimum 50 tokens
                            )
                            
                            # Get document files with same extensions as ingestion script
                            supported_extensions = {'.txt', '.md', '.pdf'}
                            doc_files = [
                                f for f in documents_dir.iterdir() 
                                if f.is_file() and f.suffix.lower() in supported_extensions
                            ]
                            
                            if doc_files:
                                # Clear existing collection first (same as ingestion script)
                                vector_store.delete_collection()
                                
                                # Process documents with same strategy as ingestion script
                                all_chunks = []
                                for doc_file in doc_files:
                                    try:
                                        chunks = processor.process_document(str(doc_file))
                                        all_chunks.extend(chunks)
                                    except Exception:
                                        continue  # Skip failed files silently
                                
                                if all_chunks:
                                    # Add chunks to vector store silently
                                    vector_store.add_chunks(all_chunks)
                                else:
                                    vector_store = None
                            else:
                                st.error(f"❌ No supported documents found in {documents_dir}")
                                vector_store = None
                        else:
                            st.error(f"❌ Documents directory not found or empty: {documents_dir}")
                            vector_store = None
                            
                    except Exception as ingest_error:
                        st.error(f"❌ Failed to auto-ingest documents: {ingest_error}")
                        st.info("� If you're deploying to cloud, ensure the 'data/documents/' directory and files are included in your deployment")
                        st.info("�🔄 Running without RAG functionality - using general AI knowledge only")
                        vector_store = None
                    
            except Exception:
                vector_store = None
                
        except Exception:
            vector_store = None
        
        chatbot = AgenticChatbot(
            ai_provider=ai_provider,
            message_handler=message_handler,
            config=config,
            vector_store=vector_store
        )
        
        return chatbot
        
    except Exception as e:
        if isinstance(e, (ConfigurationError, APIError)):
            raise
        raise ConfigurationError(f"Failed to create chatbot: {str(e)}")


def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'conversation_started' not in st.session_state:
        st.session_state.conversation_started = False


def render_chat_interface(chatbot: AgenticChatbot):
    """Render the main chat interface."""
    # Page header
    st.title("🤖 Deep AI ")
    st.markdown("Ask me anything! I'm here to help.")
    
    # Simple control panel
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_started = False
            chatbot.clear_conversation()
            st.rerun()
    
    with col2:
        # RAG status indicator
        if chatbot._agentic_agent:
            rag_status = chatbot.get_agentic_status()
            if rag_status["enabled"]:
                vector_stats = rag_status["components"].get("vector_store", {})
                total_chunks = vector_stats.get("total_chunks", 0)
                if total_chunks > 0:
                    st.success(f"📚 RAG On")
                else:
                    st.warning("📚 RAG Off")
            else:
                st.info("📚 RAG Off")
        else:
            st.info("📚 RAG Off")
    
    # Initialize conversation
    if not st.session_state.conversation_started:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        loop.run_until_complete(chatbot.start_conversation())
        st.session_state.conversation_started = True
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! How can I help you today?"
        })
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                response = loop.run_until_complete(chatbot.process_message(prompt))
                st.markdown(response)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    """Main application entry point for Streamlit Cloud."""
    # Configure page
    st.set_page_config(
        page_title="AI Chatbot",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Setup cloud environment
    setup_cloud_environment()
    
    # Initialize session state
    init_session_state()
    
    try:
        # Create chatbot instance
        chatbot = create_chatbot()
        
        # Render chat interface
        render_chat_interface(chatbot)
        
    except ConfigurationError as e:
        st.error(f"❌ Configuration Error: {e.message}")
        st.markdown("""
        **To fix this issue:**
        1. Ensure your Streamlit secrets are configured
        2. Add `GROQ_API_KEY` to your Streamlit Cloud secrets
        3. Check that all required environment variables are set
        """)
        with st.expander("🔧 Setup Instructions"):
            st.code('''
# In your Streamlit Cloud dashboard:
# Go to Settings > Secrets and add:

GROQ_API_KEY = "your_groq_api_key_here"
            ''')
        st.stop()
        
    except APIError as e:
        st.error(f"❌ API Error: {e.message}")
        st.info("Please check your API key and internet connection.")
        st.stop()
        
    except Exception as e:
        st.error(f"❌ Unexpected Error: {str(e)}")
        st.info("Please try refreshing the page. If the issue persists, check the application logs.")
        st.stop()


if __name__ == "__main__":
    main()
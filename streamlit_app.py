"""
Streamlit Cloud Deployment Entry Point

This is the main entry point that Streamlit Cloud expects for deployment.
It properly handles secrets, environment configuration, and cloud-specific requirements.
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
            
            # Debug: Check if ChromaDB directory exists
            if os.path.exists(persist_dir):
                st.info(f"📁 ChromaDB directory found at: {persist_dir}")
                # List contents for debugging
                contents = os.listdir(persist_dir) if os.path.isdir(persist_dir) else []
                st.info(f"📋 ChromaDB contents: {contents}")
            else:
                st.error(f"❌ ChromaDB directory not found at: {persist_dir}")
            
            vector_store = EnhancedChromaVectorStore(
                embedding_service=embedding_service,
                persist_directory=persist_dir
            )
            
            # Check if pre-computed embeddings exist
            stats = vector_store.get_collection_stats()
            if stats.get('total_chunks', 0) > 0:
                st.success(f"✅ RAG System: {stats['total_chunks']} document chunks loaded from pre-computed embeddings")
            else:
                st.warning("⚠️ RAG System: No pre-computed embeddings found - RAG will be disabled")
                vector_store = None
                
        except Exception as e:
            st.warning(f"⚠️ RAG System: Limited functionality - {str(e)}")
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
    st.title("🤖 Agentic AI Chatbot")
    st.markdown("""
    **Powered by Advanced RAG System** | Plan→Act→Reflect Reasoning | Document-Grounded Responses
    
    Ask me anything! I can search through documents, reason about complex queries, and provide accurate answers with source attribution.
    """)
    
    # Control panel
    col1, col2, col3, col4 = st.columns([1, 1, 1, 5])
    
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_started = False
            chatbot.clear_conversation()
            st.rerun()
    
    with col2:
        if st.button("📊 Stats", use_container_width=True):
            summary = chatbot.get_conversation_summary()
            st.info(f"💬 {summary}")
    
    with col3:
        # RAG status indicator
        if chatbot._agentic_agent:
            rag_status = chatbot.get_agentic_status()
            if rag_status["enabled"]:
                vector_stats = rag_status["components"].get("vector_store", {})
                total_chunks = vector_stats.get("total_chunks", 0)
                if total_chunks > 0:
                    st.success(f"📚 {total_chunks} docs")
                else:
                    st.warning("📚 No docs")
            else:
                st.info("📚 RAG off")
        else:
            st.info("📚 RAG off")
    
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
            "content": """Hello! I'm your **Agentic AI Assistant** with advanced reasoning capabilities. 

🧠 **What I can do:**
- **Plan→Act→Reflect** reasoning for complex queries
- Search and analyze documents with hybrid retrieval
- Provide source-attributed answers
- Handle multi-step problem solving

💬 **How to use me:**
- Ask questions about any topic
- Request document analysis or summaries
- Pose complex problems that need step-by-step reasoning

What would you like to explore today?"""
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
            with st.spinner("🧠 Processing with agentic reasoning..."):
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


def render_sidebar():
    """Render the sidebar with additional information."""
    with st.sidebar:
        st.header("🛠️ System Information")
        
        # Deployment info
        st.subheader("📡 Deployment")
        st.info("Running on Streamlit Cloud")
        
        # Features
        st.subheader("🚀 Features")
        st.markdown("""
        - **Agentic Reasoning**: Plan→Act→Reflect
        - **RAG System**: Document retrieval & analysis
        - **Hybrid Search**: Semantic + keyword matching
        - **Source Attribution**: Traceable responses
        - **Conversation Memory**: Context-aware chat
        """)
        
        # Usage tips
        st.subheader("💡 Usage Tips")
        st.markdown("""
        - Be specific in your questions
        - Ask for step-by-step explanations
        - Request document analysis
        - Use follow-up questions for clarity
        """)
        
        # Technical details
        with st.expander("🔧 Technical Details"):
            st.markdown("""
            **AI Model**: Groq (Llama-3.1-70b)
            **Embeddings**: MiniLM-L6-v2
            **Vector Store**: ChromaDB
            **Chunking**: 580-token average
            **Search**: Hybrid semantic + keyword
            """)


def main():
    """Main application entry point for Streamlit Cloud."""
    # Configure page
    st.set_page_config(
        page_title="Agentic AI Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo/chatbot',
            'Report a bug': 'https://github.com/your-repo/chatbot/issues',
            'About': 'Agentic AI Chatbot with Advanced RAG System'
        }
    )
    
    # Setup cloud environment
    setup_cloud_environment()
    
    # Initialize session state
    init_session_state()
    
    try:
        # Create chatbot instance
        chatbot = create_chatbot()
        
        # Render UI components
        render_sidebar()
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
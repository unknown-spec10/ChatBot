"""
Enhanced Streamlit Web Application Entry Point

Launch the chatbot with industry-standard RAG implementation using Streamlit.

Usage:
    streamlit run enhanced_app.py
"""

import sys
import os
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

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


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('chatbot.log'),
        ]
    )


@st.cache_resource
def create_chatbot() -> AgenticChatbot:
    """
    Create and configure the enhanced chatbot with industry-standard RAG.
    
    Returns:
        Configured enhanced chatbot instance
        
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
        
        try:
            embedding_service = MiniLMEmbeddingService(config)
            vector_store = EnhancedChromaVectorStore(
                embedding_service=embedding_service,
                persist_directory="./chroma_db"
            )
            
            stats = vector_store.get_collection_stats()
            if stats.get('total_chunks', 0) > 0:
                st.success(f"✅ Enhanced RAG loaded {stats['total_chunks']} document chunks")
            else:
                st.warning("⚠️ Enhanced RAG: No documents loaded yet")
        except Exception as e:
            st.warning(f"⚠️ Enhanced RAG unavailable: {e}")
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
        raise ConfigurationError(f"Failed to create enhanced chatbot: {str(e)}")


def main():
    """Main application entry point."""
    setup_logging()
    
    st.set_page_config(
        page_title="Agentic AI Chatbot with RAG",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    try:
        chatbot = create_chatbot()
        
        # Main chat interface
        st.title("💬 Agentic Chat with AI")
        st.markdown("Ask me anything! I use **Plan→Act→Reflect reasoning** with hybrid search for accurate, document-based answers.")
        
        # Initialize session state
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'conversation_started' not in st.session_state:
            st.session_state.conversation_started = False
        
        # Control buttons at the top
        col1, col2, col3, col4 = st.columns([1, 1, 1, 6])
        
        with col1:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_started = False
                chatbot.clear_conversation()
                st.rerun()
        
        with col2:
            if st.button("📊 Stats", use_container_width=True):
                summary = chatbot.get_conversation_summary()
                st.info(summary)
        
        with col3:
            # Show RAG status indicator
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
        
        # Start conversation
        if not st.session_state.conversation_started:
            import asyncio
            asyncio.run(chatbot.start_conversation())
            st.session_state.conversation_started = True
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Hello! I'm your agentic AI assistant with **Plan→Act→Reflect reasoning**. I can decompose complex queries, search multiple knowledge sources, and provide confident answers with source attribution. How can I help you today?"
            })
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🧠 Thinking with agentic reasoning..."):
                    import asyncio
                    response = asyncio.run(chatbot.process_message(prompt))
                    st.markdown(response)
            
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": response})
        
    except ConfigurationError as e:
        st.error(f"❌ Configuration Error: {e.message}")
        st.info("Please check your .env file and ensure GROQ_API_KEY is set.")
        st.stop()
    except APIError as e:
        st.error(f"❌ API Error: {e.message}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected Error: {str(e)}")
        st.stop()


if __name__ == "__main__":
    main()
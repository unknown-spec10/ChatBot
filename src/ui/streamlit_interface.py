"""
Streamlit Web Interface for the Chatbot.

This module provides a modern web UI using Streamlit with
chat history, document status, and session management.
"""

import asyncio
import streamlit as st
from typing import Optional
from ..interfaces.core import IUserInterface, IChatbot


class StreamlitInterface(IUserInterface):
    """
    Streamlit-based web interface implementation.
    
    Provides a modern chat interface with:
    - Persistent chat history
    - Document retrieval status
    - Session management
    - Real-time responses
    """
    
    def __init__(self, chatbot: IChatbot):
        """
        Initialize the Streamlit interface.
        
        Args:
            chatbot: The chatbot instance to interact with
        """
        self._chatbot = chatbot
        self._initialize_session_state()
    
    def _initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables."""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'conversation_started' not in st.session_state:
            st.session_state.conversation_started = False
        
        if 'chatbot_ready' not in st.session_state:
            st.session_state.chatbot_ready = False
    
    async def start(self) -> None:
        """Start the Streamlit interface."""
        st.set_page_config(
            page_title="AI Chatbot with RAG",
            page_icon="AI",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        self._render_sidebar()
        self._render_main_chat()
    
    def _render_sidebar(self) -> None:
        """Render the sidebar with info and controls."""
        with st.sidebar:
            st.title("AI Chatbot")
            st.markdown("---")
            
            st.subheader("📚 RAG Status")
            if hasattr(self._chatbot, '_vector_store') and self._chatbot._vector_store:
                try:
                    stats = self._chatbot._vector_store.get_collection_stats()
                    total_chunks = stats.get('total_chunks', 0)
                    
                    if total_chunks > 0:
                        st.success(f"[SUCCESS] {total_chunks} document chunks loaded")
                        st.info("The chatbot will answer questions based on your documents.")
                    else:
                        st.warning("⚠️ No documents loaded")
                        st.info("Add documents to enable RAG features.")
                except Exception as e:
                    st.error(f"[ERROR] RAG unavailable: {str(e)}")
            else:
                st.info("ℹ️ RAG not configured")
            
            st.markdown("---")
            
            st.subheader("ℹ️ About")
            st.markdown("""
            This chatbot uses:
            - **Groq AI** for responses
            - **MiniLM** for embeddings
            - **ChromaDB** for document search
            - **RAG** for document-based answers
            """)
            
            st.markdown("---")
            
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                self._clear_conversation()
                st.rerun()
            
            if st.button("[STATS] Show Statistics", use_container_width=True):
                self._show_statistics()
    
    def _render_main_chat(self) -> None:
        """Render the main chat interface."""
        st.title("💬 Chat with AI")
        
        if not st.session_state.conversation_started:
            asyncio.run(self._start_conversation())
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Type your message here..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = asyncio.run(self._get_response(prompt))
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    async def _start_conversation(self) -> None:
        """Start a new conversation."""
        try:
            await self._chatbot.start_conversation()
            st.session_state.conversation_started = True
            st.session_state.chatbot_ready = True
            
            welcome_message = "Hello! I'm your AI assistant. How can I help you today?"
            st.session_state.messages.append({
                "role": "assistant",
                "content": welcome_message
            })
        except Exception as e:
            st.error(f"Failed to start conversation: {str(e)}")
    
    async def _get_response(self, user_input: str) -> str:
        """
        Get a response from the chatbot.
        
        Args:
            user_input: The user's message
            
        Returns:
            The chatbot's response
        """
        try:
            response = await self._chatbot.process_message(user_input)
            return response
        except Exception as e:
            return f"[ERROR] {str(e)}"
    
    def _clear_conversation(self) -> None:
        """Clear the conversation history."""
        try:
            self._chatbot.clear_conversation()
            st.session_state.messages = []
            st.session_state.conversation_started = False
            st.success("Chat history cleared!")
        except Exception as e:
            st.error(f"Failed to clear conversation: {str(e)}")
    
    def _show_statistics(self) -> None:
        """Show conversation statistics."""
        try:
            summary = self._chatbot.get_conversation_summary()
            st.info(f"[STATS] {summary}")
        except Exception as e:
            st.error(f"Failed to get statistics: {str(e)}")
    
    async def stop(self) -> None:
        """Stop the interface and cleanup."""
        try:
            if st.session_state.conversation_started:
                await self._chatbot.end_conversation()
                st.session_state.conversation_started = False
        except Exception:
            pass


def run_streamlit_app(chatbot: IChatbot):
    """
    Run the Streamlit app.
    
    Args:
        chatbot: The chatbot instance to use
    """
    interface = StreamlitInterface(chatbot)
    asyncio.run(interface.start())

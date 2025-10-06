"""
Configuration management for the chatbot application.

This module handles loading configuration from environment variables
and Streamlit secrets for cloud deployment.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from ..interfaces.core import IConfiguration


class Configuration(IConfiguration):
    """
    Configuration implementation that loads from environment variables
    and Streamlit secrets for cloud deployment.
    
    This follows the Single Responsibility Principle by only handling
    configuration concerns.
    """
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            env_file: Optional path to .env file
        """
        # Force reload environment variables
        if env_file:
            load_dotenv(env_file, override=True)
        else:
            load_dotenv(override=True)
        
        # Try to load from Streamlit secrets if available
        self._load_streamlit_secrets()
        
        self._validate_required_config()
    
    def _load_streamlit_secrets(self) -> None:
        """Load configuration from Streamlit secrets if available."""
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                # Load API key from Streamlit secrets if not in environment
                if not os.getenv("GROQ_API_KEY") and "GROQ_API_KEY" in st.secrets:
                    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
                
                # Load other optional configs from secrets
                secret_mappings = {
                    "GROQ_MODEL": "GROQ_MODEL",
                    "MAX_TOKENS": "MAX_TOKENS", 
                    "TEMPERATURE": "TEMPERATURE",
                    "MAX_HISTORY_LENGTH": "MAX_HISTORY_LENGTH",
                    "EMBEDDING_MODEL": "MINILM_MODEL",
                    "CHUNK_SIZE": "CHUNK_SIZE",
                    "CHUNK_OVERLAP": "CHUNK_OVERLAP"
                }
                
                for secret_key, env_key in secret_mappings.items():
                    if secret_key in st.secrets and not os.getenv(env_key):
                        os.environ[env_key] = str(st.secrets[secret_key])
                        
        except ImportError:
            # Streamlit not available (e.g., running locally)
            pass
        except Exception:
            # Streamlit secrets not configured or other error
            pass
    
    def _validate_required_config(self) -> None:
        """Validate that required configuration is present."""
        if not self.get_api_key():
            raise ValueError(
                "GROQ_API_KEY is required. Please set it in:\n"
                "- Local: .env file or environment variable\n"
                "- Streamlit Cloud: App Settings > Secrets"
            )
    
    def get_api_key(self) -> str:
        """Get the Groq API key."""
        return os.getenv("GROQ_API_KEY", "")
    
    def get_model_name(self) -> str:
        """Get the model name to use."""
        return os.getenv("GROQ_MODEL", "llama3-70b-8192")
    
    def get_max_tokens(self) -> int:
        """Get the maximum tokens for responses."""
        try:
            return int(os.getenv("MAX_TOKENS", "1024"))
        except ValueError:
            return 1024
    
    def get_temperature(self) -> float:
        """Get the temperature setting for responses."""
        try:
            return float(os.getenv("TEMPERATURE", "0.7"))
        except ValueError:
            return 0.7
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the chatbot."""
        return os.getenv(
            "SYSTEM_PROMPT",
            "You are a helpful and friendly AI assistant. "
            "Provide clear, concise, and helpful responses."
        )
    
    def get_max_history_length(self) -> int:
        """Get the maximum number of messages to keep in history."""
        try:
            return int(os.getenv("MAX_HISTORY_LENGTH", "20"))
        except ValueError:
            return 20
    
    def get_embedding_provider(self) -> str:
        """Get the embedding provider to use."""
        return os.getenv("EMBEDDING_PROVIDER", "minilm")
    
    def get_minilm_model_name(self) -> str:
        """Get the MiniLM model name for local embeddings."""
        return os.getenv("MINILM_MODEL", "all-MiniLM-L6-v2")
    
    def get_chunk_size(self) -> int:
        """Get the chunk size for document processing."""
        try:
            return int(os.getenv("CHUNK_SIZE", "580"))
        except ValueError:
            return 580
    
    def get_chunk_overlap(self) -> int:
        """Get the chunk overlap for document processing."""
        try:
            return int(os.getenv("CHUNK_OVERLAP", "100"))
        except ValueError:
            return 100
    
    def get_max_chunks_retrieved(self) -> int:
        """Get the maximum number of chunks to retrieve for RAG."""
        try:
            return int(os.getenv("MAX_CHUNKS_RETRIEVED", "5"))
        except ValueError:
            return 5
    
    def get_confidence_threshold(self) -> float:
        """Get the confidence threshold for escalation."""
        try:
            return float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
        except ValueError:
            return 0.7
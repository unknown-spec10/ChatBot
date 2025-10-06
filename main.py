"""
Main application entry point for the Simple Chatbot.

This module wires together all the components following
dependency injection principles and SOLID design.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

if 'GROQ_API_KEY' in os.environ:
    del os.environ['GROQ_API_KEY']

from src.config.settings import Configuration
from src.services.groq_provider import GroqAIProvider
from src.services.message_handler import ConversationHandler
from src.services.chatbot import AgenticChatbot
from src.ui.cli_interface import CLIInterface
from src.exceptions.chatbot_exceptions import (
    ChatbotError, 
    ConfigurationError, 
    APIError
)
from src.rag.minilm_embeddings import MiniLMEmbeddingService
from src.rag.vector_store import EnhancedChromaVectorStore


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('chatbot.log'),
            logging.StreamHandler()
        ]
    )


async def create_chatbot() -> AgenticChatbot:
    """
    Create and configure the chatbot with all dependencies.
    
    This function demonstrates dependency injection and
    follows the SOLID principles by composing the application
    from interfaces and implementations.
    
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
        
        print("[LOADING] Loading document retrieval system...")
        try:
            embedding_service = MiniLMEmbeddingService(config)
            vector_store = EnhancedChromaVectorStore(
                embedding_service=embedding_service,
                persist_directory="./chroma_db"
            )
            print("[SUCCESS] Document retrieval ready")
            stats = vector_store.get_collection_stats()
            if stats.get('total_chunks', 0) > 0:
                print(f"[SUCCESS] Found {stats['total_chunks']} document chunks")
            else:
                print("[WARNING] No documents found. Run: python ingest_documents.py")
        except Exception as e:
            print(f"[WARNING] Document retrieval unavailable: {e}")
            vector_store = None
        
        chatbot = AgenticChatbot(
            ai_provider=ai_provider,
            message_handler=message_handler,
            config=config,
            vector_store=vector_store
        )
        
        if not await chatbot.validate_setup():
            raise ConfigurationError(
                "Chatbot setup validation failed. Please check your configuration."
            )
        
        return chatbot
        
    except Exception as e:
        if isinstance(e, (ConfigurationError, APIError)):
            raise
        raise ConfigurationError(f"Failed to create chatbot: {str(e)}")


async def main() -> None:
    """
    Main application entry point.
    
    This function sets up logging, creates the chatbot,
    and starts the CLI interface.
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting Agentic Chatbot application")
        chatbot = await create_chatbot()
        logger.info("Chatbot created successfully")
        cli = CLIInterface(chatbot)
        await cli.start()
        
    except ConfigurationError as e:
        print(f"❌ Configuration Error: {e.message}")
        if e.details:
            print(f"Details: {e.details}")
        print("\\nPlease check your .env file and ensure GROQ_API_KEY is set correctly.")
        sys.exit(1)
        
    except APIError as e:
        print(f"❌ API Error: {e.message}")
        if e.details:
            print(f"Details: {e.details}")
        print("\\nPlease check your internet connection and API key.")
        sys.exit(1)
        
    except ChatbotError as e:
        print(f"❌ Chatbot Error: {e.message}")
        if e.details:
            print(f"Details: {e.details}")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\\n👋 Application interrupted by user. Goodbye!")
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        print(f"❌ Unexpected error: {str(e)}")
        print("Check the log file (chatbot.log) for more details.")
        sys.exit(1)
        
    finally:
        logger.info("Agentic Chatbot application ended")


if __name__ == "__main__":
    asyncio.run(main())
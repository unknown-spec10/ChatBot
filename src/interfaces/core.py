"""
Core interfaces for the chatbot application following SOLID principles.

These interfaces define contracts for different components:
- AI Provider: Interface for different AI services (Groq, OpenAI, etc.)
- Message Handler: Interface for managing chat messages
- Configuration: Interface for configuration management
- User Interface: Interface for different UI implementations
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from ..models.message import Message


class IAIProvider(ABC):
    """
    Interface for AI service providers.
    
    This follows the Dependency Inversion Principle by allowing
    different AI providers to be used interchangeably.
    """
    
    @abstractmethod
    async def generate_response(self, messages: List[Message]) -> str:
        """Generate a response based on conversation history."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the AI provider is available and configured."""
        pass


class IMessageHandler(ABC):
    """
    Interface for managing conversation messages.
    
    This follows the Single Responsibility Principle by separating
    message management from AI generation.
    """
    
    @abstractmethod
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        pass
    
    @abstractmethod
    def get_messages(self) -> List[Message]:
        """Get all messages in the conversation."""
        pass
    
    @abstractmethod
    def clear_history(self) -> None:
        """Clear conversation history."""
        pass
    
    @abstractmethod
    def get_recent_messages(self, count: int = 10) -> List[Message]:
        """Get recent messages for context."""
        pass


class IConfiguration(ABC):
    """
    Interface for configuration management.
    
    This follows the Interface Segregation Principle by providing
    only the configuration methods needed.
    """
    
    @abstractmethod
    def get_api_key(self) -> str:
        """Get the API key for the AI provider."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name to use."""
        pass
    
    @abstractmethod
    def get_max_tokens(self) -> int:
        """Get the maximum tokens for responses."""
        pass
    
    @abstractmethod
    def get_temperature(self) -> float:
        """Get the temperature setting for responses."""
        pass


class IUserInterface(ABC):
    """
    Interface for user interaction.
    
    This follows the Open/Closed Principle by allowing different
    UI implementations (CLI, Web, GUI) without changing core logic.
    """
    
    @abstractmethod
    async def get_user_input(self) -> str:
        """Get input from the user."""
        pass
    
    @abstractmethod
    async def display_message(self, message: str, sender: str = "Assistant") -> None:
        """Display a message to the user."""
        pass
    
    @abstractmethod
    async def display_error(self, error: str) -> None:
        """Display an error message to the user."""
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """Start the user interface."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the user interface."""
        pass


class IChatbot(ABC):
    """
    Interface for the main chatbot engine.
    
    This follows the Dependency Inversion Principle by depending
    on abstractions rather than concrete implementations.
    """
    
    @abstractmethod
    async def process_message(self, user_input: str) -> str:
        """Process a user message and return a response."""
        pass
    
    @abstractmethod
    async def start_conversation(self) -> None:
        """Start a new conversation."""
        pass
    
    @abstractmethod
    async def end_conversation(self) -> None:
        """End the current conversation."""
        pass
"""
Message model for the chatbot application.

This model represents a single message in a conversation,
following the Single Responsibility Principle.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class MessageRole(Enum):
    """Enumeration for message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """
    Represents a message in the conversation.
    
    This is a simple data class that encapsulates message data
    without any business logic, following SOLID principles.
    """
    role: MessageRole
    content: str
    timestamp: datetime
    metadata: Optional[dict] = None
    
    def __post_init__(self):
        """Initialize default values after object creation."""
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> dict:
        """Convert message to dictionary format for API calls."""
        return {
            "role": self.role.value,
            "content": self.content
        }
    
    def __str__(self) -> str:
        """String representation of the message."""
        return f"[{self.role.value.upper()}] {self.content}"
    
    @classmethod
    def create_user_message(cls, content: str) -> "Message":
        """Factory method to create a user message."""
        return cls(
            role=MessageRole.USER,
            content=content,
            timestamp=datetime.now()
        )
    
    @classmethod
    def create_assistant_message(cls, content: str) -> "Message":
        """Factory method to create an assistant message."""
        return cls(
            role=MessageRole.ASSISTANT,
            content=content,
            timestamp=datetime.now()
        )
    
    @classmethod
    def create_system_message(cls, content: str) -> "Message":
        """Factory method to create a system message."""
        return cls(
            role=MessageRole.SYSTEM,
            content=content,
            timestamp=datetime.now()
        )
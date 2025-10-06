"""
Message handling implementation for the chatbot.

This module provides concrete implementation of message management
following the Single Responsibility Principle.
"""

from typing import List
from ..interfaces.core import IMessageHandler
from ..models.message import Message, MessageRole


class ConversationHandler(IMessageHandler):
    """
    Handles conversation messages and history.
    
    This class is responsible only for managing the conversation
    state and messages, following SRP.
    """
    
    def __init__(self, max_history_length: int = 20):
        """
        Initialize the conversation handler.
        
        Args:
            max_history_length: Maximum number of messages to keep in history
        """
        self._messages: List[Message] = []
        self._max_history_length = max_history_length
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self._messages.append(message)
        self._trim_history()
    
    def get_messages(self) -> List[Message]:
        """Get all messages in the conversation."""
        return self._messages.copy()
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self._messages.clear()
    
    def get_recent_messages(self, count: int = 10) -> List[Message]:
        """Get recent messages for context."""
        return self._messages[-count:] if self._messages else []
    
    def _trim_history(self) -> None:
        """Trim history to maintain maximum length."""
        if len(self._messages) > self._max_history_length:
            # Keep system messages and trim user/assistant messages
            system_messages = [msg for msg in self._messages if msg.role == MessageRole.SYSTEM]
            other_messages = [msg for msg in self._messages if msg.role != MessageRole.SYSTEM]
            
            # Keep only the most recent non-system messages
            recent_others = other_messages[-(self._max_history_length - len(system_messages)):]
            
            self._messages = system_messages + recent_others
    
    def get_messages_for_api(self) -> List[dict]:
        """Get messages formatted for API calls."""
        return [msg.to_dict() for msg in self._messages]
    
    def add_system_message(self, content: str) -> None:
        """Add a system message to the conversation."""
        system_message = Message.create_system_message(content)
        self.add_message(system_message)
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        user_message = Message.create_user_message(content)
        self.add_message(user_message)
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation."""
        assistant_message = Message.create_assistant_message(content)
        self.add_message(assistant_message)
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation."""
        if not self._messages:
            return "No conversation history."
        
        total_messages = len(self._messages)
        user_messages = len([msg for msg in self._messages if msg.role == MessageRole.USER])
        assistant_messages = len([msg for msg in self._messages if msg.role == MessageRole.ASSISTANT])
        
        return (
            f"Conversation Summary:\n"
            f"- Total messages: {total_messages}\n"
            f"- User messages: {user_messages}\n"
            f"- Assistant messages: {assistant_messages}\n"
            f"- Started: {self._messages[0].timestamp.strftime('%Y-%m-%d %H:%M:%S') if self._messages else 'N/A'}"
        )
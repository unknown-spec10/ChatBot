"""
Custom exceptions for the chatbot application.

This module defines specific exceptions for different error scenarios,
following the principle of explicit error handling.
"""


class ChatbotError(Exception):
    """Base exception for all chatbot-related errors."""
    
    def __init__(self, message: str, details: str = None):
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            details: Additional technical details
        """
        super().__init__(message)
        self.message = message
        self.details = details or ""
    
    def __str__(self) -> str:
        """String representation of the error."""
        if self.details:
            return f"{self.message} - {self.details}"
        return self.message


class ConfigurationError(ChatbotError):
    """Exception raised for configuration-related errors."""
    pass


class APIError(ChatbotError):
    """Exception raised for API-related errors."""
    pass


class ValidationError(ChatbotError):
    """Exception raised for input validation errors."""
    pass


class ConversationError(ChatbotError):
    """Exception raised for conversation handling errors."""
    pass


class UIError(ChatbotError):
    """Exception raised for user interface errors."""
    pass
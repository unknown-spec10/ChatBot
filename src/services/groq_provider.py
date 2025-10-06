"""
Groq AI Provider implementation.

This module implements the IAIProvider interface for Groq API,
following the Dependency Inversion Principle.
"""

import asyncio
from typing import List
from groq import Groq
from ..interfaces.core import IAIProvider, IConfiguration
from ..models.message import Message
from ..exceptions.chatbot_exceptions import APIError, ConfigurationError


class GroqAIProvider(IAIProvider):
    """
    Groq AI service implementation.
    
    This class handles communication with the Groq API while
    implementing the IAIProvider interface for interchangeability.
    """
    
    def __init__(self, config: IConfiguration):
        """
        Initialize the Groq AI provider.
        
        Args:
            config: Configuration object that provides API settings
        """
        self._config = config
        api_key = config.get_api_key()
        
        # Validate API key format
        if not api_key or not api_key.startswith('gsk_') or len(api_key) < 40:
            raise ConfigurationError(
                "Invalid Groq API key format. "
                "Please ensure your GROQ_API_KEY in .env file is a valid key from https://console.groq.com/keys"
            )
        
        try:
            self._client = Groq(api_key=api_key)
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Groq client: {str(e)}")
    
    async def generate_response(self, messages: List[Message]) -> str:
        """
        Generate a response using Groq API.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Generated response string
            
        Raises:
            APIError: If the API call fails
        """
        try:
            # Convert messages to API format
            api_messages = [msg.to_dict() for msg in messages]
            
            # Make the API call in a thread pool to avoid blocking
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self._make_api_call,
                api_messages
            )
            
            return response.choices[0].message.content or ""
            
        except Exception as e:
            raise APIError(f"Failed to generate response: {str(e)}")
    
    def _make_api_call(self, messages):
        """
        Make the actual API call to Groq.
        
        This is separated to allow for easier testing and
        to run in a thread pool.
        """
        return self._client.chat.completions.create(
            model=self._config.get_model_name(),
            messages=messages,
            max_tokens=self._config.get_max_tokens(),
            temperature=self._config.get_temperature(),
            stream=False
        )
    
    def is_available(self) -> bool:
        """
        Check if the Groq provider is available and configured.
        
        Returns:
            True if the provider is ready to use
        """
        try:
            # Basic check for API key presence
            api_key = self._config.get_api_key()
            return bool(api_key and api_key.strip())
        except Exception:
            return False
    
    async def validate_connection(self) -> bool:
        """
        Validate the connection to Groq API.
        
        Returns:
            True if connection is successful
        """
        try:
            # Test with a simple message
            test_messages = [Message.create_user_message("Hello")]
            response = await self.generate_response(test_messages)
            return bool(response and response.strip())
        except Exception:
            return False
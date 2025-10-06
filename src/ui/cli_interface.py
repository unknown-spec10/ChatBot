"""
Command Line Interface for the chatbot.

This module provides a simple CLI implementation following
the Interface Segregation Principle.
"""

import asyncio
import sys
from typing import Optional
from ..interfaces.core import IUserInterface, IChatbot
from ..exceptions.chatbot_exceptions import UIError


class CLIInterface(IUserInterface):
    """
    Command Line Interface implementation.
    
    This class handles user interaction through the command line
    while implementing the IUserInterface for flexibility.
    """
    
    def __init__(self, chatbot: IChatbot):
        """
        Initialize the CLI interface.
        
        Args:
            chatbot: The chatbot instance to interact with
        """
        self._chatbot = chatbot
        self._is_running = False
        self._commands = {
            '/help': self._show_help,
            '/clear': self._clear_conversation,
            '/summary': self._show_summary,
            '/quit': self._quit,
            '/exit': self._quit,
        }
    
    async def get_user_input(self) -> str:
        """Get input from the user."""
        try:
            # Run input in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            user_input = await loop.run_in_executor(None, input, "You: ")
            return user_input.strip()
        except (EOFError, KeyboardInterrupt):
            return "/quit"
        except Exception as e:
            raise UIError(f"Failed to get user input: {str(e)}")
    
    async def display_message(self, message: str, sender: str = "Assistant") -> None:
        """Display a message to the user."""
        try:
            print(f"{sender}: {message}")
        except Exception as e:
            raise UIError(f"Failed to display message: {str(e)}")
    
    async def display_error(self, error: str) -> None:
        """Display an error message to the user."""
        try:
            print(f"[ERROR] {error}")
        except Exception as e:
            print(f"Critical error displaying error message: {str(e)}")
    
    async def start(self) -> None:
        """Start the CLI interface."""
        try:
            self._is_running = True
            await self._display_welcome()
            await self._start_conversation()
            await self._main_loop()
        except KeyboardInterrupt:
            await self.display_message("\\nGoodbye!")
        except Exception as e:
            await self.display_error(f"CLI interface error: {str(e)}")
        finally:
            await self.stop()
    
    async def stop(self) -> None:
        """Stop the CLI interface."""
        self._is_running = False
        if self._chatbot.is_conversation_active():
            await self._chatbot.end_conversation()
    
    async def _display_welcome(self) -> None:
        """Display welcome message and instructions."""
        welcome_msg = """
[CHATBOT] Simple Chatbot (Powered by Groq)
==========================================

Welcome! I'm here to help you with any questions or conversations.

Available commands:
• /help     - Show this help message
• /clear    - Clear conversation history
• /summary  - Show conversation summary
• /quit     - Exit the chatbot

Just type your message and press Enter to start chatting!
"""
        print(welcome_msg)
    
    async def _start_conversation(self) -> None:
        """Start the chatbot conversation."""
        try:
            await self._chatbot.start_conversation()
            await self.display_message("Hello! How can I help you today?")
        except Exception as e:
            await self.display_error(f"Failed to start conversation: {str(e)}")
            raise
    
    async def _main_loop(self) -> None:
        """Main interaction loop."""
        while self._is_running:
            try:
                user_input = await self.get_user_input()
                
                if not user_input:
                    continue
                
                # Check for commands
                if user_input.startswith('/'):
                    await self._handle_command(user_input)
                else:
                    await self._process_user_message(user_input)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                await self.display_error(f"Unexpected error: {str(e)}")
    
    async def _handle_command(self, command: str) -> None:
        """Handle user commands."""
        command_parts = command.split()
        command_name = command_parts[0].lower()
        
        if command_name in self._commands:
            await self._commands[command_name]()
        else:
            await self.display_error(f"Unknown command: {command_name}. Type /help for available commands.")
    
    async def _process_user_message(self, message: str) -> None:
        """Process a regular user message."""
        try:
            # Show typing indicator
            await self.display_message("thinking...", "Assistant")
            
            # Get response from chatbot
            response = await self._chatbot.process_message(message)
            
            # Clear the thinking indicator (simple approach)
            print("\\033[A\\033[K", end="")  # Move up and clear line
            
            # Display the actual response
            await self.display_message(response)
            
        except Exception as e:
            print("\\033[A\\033[K", end="")  # Clear thinking indicator
            await self.display_error(f"Failed to process message: {str(e)}")
    
    async def _show_help(self) -> None:
        """Show help information."""
        help_msg = """
Available Commands:
==================
• /help     - Show this help message
• /clear    - Clear conversation history (keeps system prompt)
• /summary  - Show conversation summary with statistics
• /quit     - Exit the chatbot application

Simply type your message to chat with the AI assistant!
"""
        print(help_msg)
    
    async def _clear_conversation(self) -> None:
        """Clear conversation history."""
        try:
            self._chatbot.clear_conversation()
            await self.display_message("Conversation history cleared! How can I help you?")
        except Exception as e:
            await self.display_error(f"Failed to clear conversation: {str(e)}")
    
    async def _show_summary(self) -> None:
        """Show conversation summary."""
        try:
            summary = self._chatbot.get_conversation_summary()
            print(f"\\n{summary}\\n")
        except Exception as e:
            await self.display_error(f"Failed to get conversation summary: {str(e)}")
    
    async def _quit(self) -> None:
        """Quit the application."""
        await self.display_message("Thank you for using the chatbot! Goodbye! 👋")
        self._is_running = False
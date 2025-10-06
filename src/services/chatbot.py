"""
Agentic Chatbot Service with Plan → Act → Reflect RAG.

This module integrates the Agentic RAG Agent with the chatbot interface,
providing sophisticated multi-step reasoning and document retrieval.
"""

import logging
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from ..interfaces.core import IChatbot, IAIProvider, IMessageHandler, IConfiguration
from ..models.message import Message, MessageRole
from ..exceptions.chatbot_exceptions import ConversationError, ValidationError
from ..services.agentic_rag import AgenticRAGAgent
from ..rag.faiss_hybrid_search import FaissHybridRetriever, convert_faiss_to_retrieval_results
from ..rag.reranker import AdaptiveReranker

if TYPE_CHECKING:
    from ..rag.vector_store import EnhancedChromaVectorStore


class AgenticChatbot(IChatbot):
    """
    Agentic chatbot implementation with sophisticated RAG.
    
    Features:
    - Plan → Act → Reflect reasoning cycles
    - Query decomposition and intent routing
    - Hybrid search (semantic + keyword)
    - Cross-encoder re-ranking
    - LLM-as-a-Judge confidence assessment
    - Adaptive response generation
    - Source attribution and verification
    """
    
    def __init__(
        self,
        ai_provider: IAIProvider,
        message_handler: IMessageHandler,
        config: IConfiguration,
        vector_store: Optional['EnhancedChromaVectorStore'] = None
    ):
        """
        Initialize the agentic chatbot.
        
        Args:
            ai_provider: AI service provider (e.g., Groq)
            message_handler: Message management service
            config: Configuration service
            vector_store: Optional enhanced vector store for RAG functionality
        """
        self._ai_provider = ai_provider
        self._message_handler = message_handler
        self._config = config
        self._vector_store = vector_store
        self._is_conversation_active = False
        
        # Set up logging
        self._logger = logging.getLogger(__name__)
        
        # Initialize agentic components
        if self._vector_store:
            self._hybrid_retriever = FaissHybridRetriever(
                vector_store=vector_store,
                embedding_dimension=384,  # MiniLM dimension
                dense_weight=0.7,
                sparse_weight=0.3
            )
            
            self._agentic_agent = AgenticRAGAgent(
                ai_provider=ai_provider,
                vector_store=vector_store,
                hybrid_retriever=self._hybrid_retriever,
                max_reflection_cycles=2
            )
            
            self._adaptive_reranker = AdaptiveReranker()
            
            stats = self._vector_store.get_collection_stats()
            self._logger.info(
                f"[AGENTIC] RAG initialized with {stats['total_chunks']} chunks"
            )
        else:
            self._agentic_agent = None
            self._hybrid_retriever = None
            self._adaptive_reranker = None
            self._logger.info("[AGENTIC] Chatbot initialized without RAG")
        
        # Validate dependencies
        self._validate_dependencies()
    
    def _validate_dependencies(self) -> None:
        """Validate that all dependencies are properly configured."""
        if not self._ai_provider.is_available():
            raise ConversationError(
                "AI provider is not available. Please check your configuration."
            )
    
    async def process_message(self, user_input: str) -> str:
        """
        Process a user message with agentic RAG reasoning.
        
        Args:
            user_input: The user's input message
            
        Returns:
            The chatbot's response
            
        Raises:
            ValidationError: If input is invalid
            ConversationError: If processing fails
        """
        try:
            # Validate input
            if not user_input or not user_input.strip():
                raise ValidationError("User input cannot be empty")
            
            # Add user message to conversation
            self._message_handler.add_user_message(user_input.strip())
            
            # Process with agentic RAG if available
            if self._agentic_agent:
                response_data = await self._process_with_agentic_rag(user_input.strip())
                response = response_data["response"]
                
                # Log agentic reasoning
                self._log_agentic_reasoning(response_data)
            else:
                response = await self._process_without_rag(user_input.strip())
            
            if not response or not response.strip():
                response = "I'm sorry, I couldn't generate a response. Please try again."
            
            # Add assistant response to conversation
            self._message_handler.add_assistant_message(response.strip())
            
            self._logger.info(f"Processed message successfully. Response length: {len(response)}")
            
            return response.strip()
            
        except ValidationError:
            raise
        except Exception as e:
            self._logger.error(f"Error processing message: {str(e)}")
            raise ConversationError(f"Failed to process message: {str(e)}")
    
    async def _process_with_agentic_rag(self, user_input: str) -> Dict[str, Any]:
        """Process message using the full agentic RAG pipeline."""
        try:
            self._logger.info(f"[START] Starting agentic RAG processing for: {user_input[:50]}...")
            
            # Run the complete agentic RAG pipeline
            response_data = await self._agentic_agent.process_query(user_input)
            
            return response_data
            
        except Exception as e:
            self._logger.error(f"Agentic RAG processing failed: {e}")
            # Fallback to simple processing
            response = await self._process_without_rag(user_input)
            return {
                "response": response,
                "metadata": {"fallback": True, "error": str(e)},
                "agent_reasoning": []
            }
    
    async def _process_without_rag(self, user_input: str) -> str:
        """Process message without RAG using general knowledge."""
        self._logger.info("Processing without RAG - using general knowledge")
        
        # Get conversation context
        messages = self._message_handler.get_messages()
        
        # Use higher temperature for general responses
        return await self._ai_provider.generate_response(messages)
    
    def _log_agentic_reasoning(self, response_data: Dict[str, Any]) -> None:
        """Log the agent's reasoning process for debugging."""
        metadata = response_data.get("metadata", {})
        agent_steps = response_data.get("agent_reasoning", [])
        
        self._logger.info(
            f"[REASONING] Agentic reasoning: {metadata.get('agent_steps', 0)} steps, "
            f"strategy: {metadata.get('generation_strategy', 'unknown')}, "
            f"confidence: {metadata.get('confidence_assessment', {}).get('overall_confidence', 0):.1f}"
        )
        
        # Log detailed steps in debug mode
        for i, step in enumerate(agent_steps):
            self._logger.debug(
                f"Step {i+1}: {step.thought_type.value} "
                f"(confidence: {step.confidence_score:.2f}) - {step.reasoning}"
            )
    
    async def start_conversation(self) -> None:
        """
        Start a new conversation.
        
        This initializes the conversation with a system prompt.
        """
        try:
            if self._is_conversation_active:
                self._logger.warning("Conversation already active")
                return
            
            # Clear any existing conversation
            self._message_handler.clear_history()
            
            # Add enhanced system prompt for agentic behavior
            if self._agentic_agent:
                system_prompt = self._get_agentic_system_prompt()
            else:
                system_prompt = self._config.get_system_prompt()
            
            self._message_handler.add_system_message(system_prompt)
            
            self._is_conversation_active = True
            self._logger.info("Agentic conversation started successfully")
            
        except Exception as e:
            self._logger.error(f"Error starting conversation: {str(e)}")
            raise ConversationError(f"Failed to start conversation: {str(e)}")
    
    def _get_agentic_system_prompt(self) -> str:
        """Get system prompt for agentic behavior."""
        base_prompt = self._config.get_system_prompt()
        
        agentic_addition = """

AGENTIC CAPABILITIES:
You have access to an advanced Agentic RAG system with the following capabilities:

1. PLAN → ACT → REFLECT: You use multi-step reasoning cycles to analyze queries, search for information, assess confidence, and reflect on results.

2. QUERY PROCESSING: Complex queries are automatically decomposed into simpler sub-queries for better coverage.

3. HYBRID SEARCH: You combine semantic vector search with keyword/BM25 search for maximum recall and precision.

4. CROSS-ENCODER RE-RANKING: Retrieved documents are re-ranked using specialized models for optimal relevance.

5. CONFIDENCE ASSESSMENT: You use LLM-as-a-Judge to evaluate the quality of retrieved information and adapt your response accordingly.

6. SOURCE ATTRIBUTION: You provide clear source citations for document-based information using the format [Source X: document_name].

7. ADAPTIVE RESPONSES: Your response style adapts based on confidence levels:
   - High confidence (≥4.0): Factual, low temperature, strict document adherence
   - Medium confidence (2.0-4.0): Hybrid approach, balanced temperature
   - Low confidence (<2.0): General knowledge, higher temperature

When answering questions, you will automatically engage the appropriate level of reasoning and provide transparent information about your confidence and sources.
"""
        
        return base_prompt + agentic_addition
    
    async def end_conversation(self) -> None:
        """End the current conversation."""
        try:
            self._is_conversation_active = False
            self._logger.info("Agentic conversation ended")
            
        except Exception as e:
            self._logger.error(f"Error ending conversation: {str(e)}")
            raise ConversationError(f"Failed to end conversation: {str(e)}")
    
    def is_conversation_active(self) -> bool:
        """Check if a conversation is currently active."""
        return self._is_conversation_active
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation."""
        if hasattr(self._message_handler, 'get_conversation_summary'):
            summary = self._message_handler.get_conversation_summary()
        else:
            messages = self._message_handler.get_messages()
            summary = f"Agentic conversation has {len(messages)} messages"
        
        # Add agentic capabilities info
        if self._agentic_agent:
            return f"{summary} | Agentic RAG: Plan→Act→Reflect enabled"
        else:
            return f"{summary} | Agentic RAG: disabled"
    
    def clear_conversation(self) -> None:
        """Clear the current conversation history."""
        try:
            self._message_handler.clear_history()
            # Re-add system prompt if conversation is active
            if self._is_conversation_active:
                if self._agentic_agent:
                    system_prompt = self._get_agentic_system_prompt()
                else:
                    system_prompt = self._config.get_system_prompt()
                self._message_handler.add_system_message(system_prompt)
            
            self._logger.info("Agentic conversation history cleared")
            
        except Exception as e:
            self._logger.error(f"Error clearing conversation: {str(e)}")
            raise ConversationError(f"Failed to clear conversation: {str(e)}")
    
    async def validate_setup(self) -> bool:
        """
        Validate that the agentic chatbot is properly set up.
        
        Returns:
            True if setup is valid
        """
        try:
            # Check AI provider
            if not self._ai_provider.is_available():
                self._logger.error("AI provider not available")
                return False
            
            # Check agentic components
            if self._agentic_agent:
                # Validate vector store
                stats = self._vector_store.get_collection_stats()
                if stats["total_chunks"] == 0:
                    self._logger.warning("Vector store has no documents")
                else:
                    self._logger.info(f"Vector store validated with {stats['total_chunks']} chunks")
                
                # Validate hybrid retriever
                hybrid_stats = self._hybrid_retriever.get_stats()
                self._logger.info(
                    f"FAISS hybrid search: {hybrid_stats['total_documents']} docs indexed, "
                    f"dense: {hybrid_stats['dense_index_size']}, sparse: {hybrid_stats['sparse_index_size']}"
                )
                
                # Validate reranker
                reranker_info = self._adaptive_reranker.cross_encoder.get_model_info()
                self._logger.info(
                    f"Cross-encoder: {reranker_info['model_name']} "
                    f"({'loaded' if reranker_info['model_loaded'] else 'fallback'})"
                )
            
            # Test with a simple message if provider supports it
            if hasattr(self._ai_provider, 'validate_connection'):
                return await self._ai_provider.validate_connection()
            
            return True
            
        except Exception as e:
            self._logger.error(f"Agentic setup validation failed: {str(e)}")
            return False
    
    def get_agentic_status(self) -> Dict[str, Any]:
        """Get detailed status of agentic capabilities."""
        if not self._agentic_agent:
            return {
                "enabled": False,
                "components": {},
                "capabilities": []
            }
        
        # Get component status
        vector_stats = self._vector_store.get_collection_stats()
        hybrid_stats = self._hybrid_retriever.get_stats()
        reranker_info = self._adaptive_reranker.cross_encoder.get_model_info()
        
        return {
            "enabled": True,
            "components": {
                "vector_store": {
                    "total_chunks": vector_stats["total_chunks"],
                    "embedding_dimension": vector_stats.get("embedding_dimension", "unknown")
                },
                "hybrid_search": {
                    "total_documents": hybrid_stats["total_documents"],
                    "dense_index_size": hybrid_stats["dense_index_size"],
                    "sparse_index_size": hybrid_stats["sparse_index_size"],
                    "dense_weight": hybrid_stats["dense_weight"],
                    "sparse_weight": hybrid_stats["sparse_weight"],
                    "indices_initialized": hybrid_stats["indices_initialized"]
                },
                "cross_encoder": {
                    "model": reranker_info["model_name"],
                    "loaded": reranker_info["model_loaded"],
                    "fallback": reranker_info["fallback_enabled"]
                }
            },
            "capabilities": [
                "Plan → Act → Reflect Reasoning",
                "Query Decomposition & Intent Routing", 
                "Hybrid Search (Semantic + Keyword)",
                "Cross-Encoder Re-ranking",
                "LLM-as-a-Judge Confidence Assessment",
                "Adaptive Response Generation",
                "Source Attribution & Verification"
            ]
        }
    
    async def explain_reasoning(self, query: str) -> Dict[str, Any]:
        """
        Get detailed explanation of how the agent would process a query.
        
        Args:
            query: Query to analyze
            
        Returns:
            Dictionary with detailed reasoning explanation
        """
        if not self._agentic_agent:
            return {"error": "Agentic RAG not available"}
        
        try:
            # Process query and get full reasoning
            response_data = await self._agentic_agent.process_query(query)
            
            # Extract reasoning steps
            reasoning_steps = response_data.get("agent_reasoning", [])
            
            explanation = {
                "query": query,
                "reasoning_steps": reasoning_steps,
                "metadata": response_data.get("metadata", {}),
                "final_response": response_data.get("response", ""),
                "summary": self._summarize_reasoning(reasoning_steps)
            }
            
            return explanation
            
        except Exception as e:
            return {"error": f"Failed to explain reasoning: {e}"}
    
    def _summarize_reasoning(self, steps: List[Dict[str, Any]]) -> str:
        """Summarize the agent's reasoning process."""
        if not steps:
            return "No reasoning steps recorded"
        
        summary_parts = []
        
        for step in steps:
            thought_type = step.get("thought_type", "unknown")
            confidence = step.get("confidence", 0)
            reasoning = step.get("reasoning", "")
            
            summary_parts.append(
                f"{thought_type.replace('_', ' ').title()}: {reasoning} "
                f"(confidence: {confidence:.2f})"
            )
        
        return " → ".join(summary_parts)
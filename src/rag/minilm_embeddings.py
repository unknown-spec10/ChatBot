"""
Local MiniLM embedding service using sentence-transformers.

This module provides embedding functionality using the local
all-MiniLM-L6-v2 model, which is lightweight and runs without API calls.
"""

import logging
import os
from typing import List, Optional
import numpy as np
from ..interfaces.core import IConfiguration
from ..exceptions.chatbot_exceptions import APIError, ConfigurationError


class MiniLMEmbeddingService:
    """
    Local MiniLM embedding service for offline RAG implementation.
    
    This class provides text embedding functionality using the
    sentence-transformers library with the all-MiniLM-L6-v2 model,
    which is lightweight and suitable for local deployment.
    """
    
    def __init__(self, config: IConfiguration, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the MiniLM embedding service.
        
        Args:
            config: Configuration object
            model_name: Name of the sentence-transformer model to use
        """
        self.config = config
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        self.model = None
        
        # Initialize the model
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            self.logger.info(f"Loading MiniLM model: {self.model_name}")
            
            # Load the model (it will download if not already cached)
            self.model = SentenceTransformer(self.model_name)
            
            self.logger.info(f"Successfully loaded MiniLM model: {self.model_name}")
            
        except ImportError:
            raise ConfigurationError(
                "sentence-transformers not installed. "
                "Install it with: pip install sentence-transformers torch"
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to load MiniLM model: {str(e)}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
            
        Raises:
            APIError: If embedding generation fails
        """
        try:
            # Clean and prepare text
            cleaned_text = self._prepare_text(text)
            
            if not cleaned_text.strip():
                raise ValueError("Text is empty after cleaning")
            
            # Generate embedding
            embedding = self.model.encode(cleaned_text, convert_to_tensor=False)
            
            # Convert to list if it's a numpy array
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {str(e)}")
            raise APIError(f"Embedding generation failed: {str(e)}")
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embeddings for a query text.
        
        For MiniLM, query and document embeddings use the same process.
        
        Args:
            query: Query text to embed
            
        Returns:
            List of embedding values
        """
        return self.embed_text(query)
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Clean all texts
            cleaned_texts = [self._prepare_text(text) for text in texts]
            
            # Filter out empty texts and keep track of indices
            valid_texts = []
            valid_indices = []
            
            for i, text in enumerate(cleaned_texts):
                if text.strip():
                    valid_texts.append(text)
                    valid_indices.append(i)
            
            # Generate embeddings for valid texts
            if valid_texts:
                embeddings = self.model.encode(valid_texts, convert_to_tensor=False)
                
                # Convert to list format
                if isinstance(embeddings, np.ndarray):
                    embeddings = embeddings.tolist()
            else:
                embeddings = []
            
            # Create result list with empty embeddings for invalid texts
            result = []
            embedding_idx = 0
            
            for i in range(len(texts)):
                if i in valid_indices:
                    result.append(embeddings[embedding_idx])
                    embedding_idx += 1
                else:
                    result.append([])
                    self.logger.warning(f"Empty embedding for text {i}")
            
            self.logger.info(f"Generated {len(embeddings)} embeddings from {len(texts)} texts")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate batch embeddings: {str(e)}")
            raise APIError(f"Batch embedding generation failed: {str(e)}")
    
    def _prepare_text(self, text: str) -> str:
        """
        Prepare text for embedding generation.
        
        Args:
            text: Raw text to prepare
            
        Returns:
            Cleaned and prepared text
        """
        if not text:
            return ""
        
        # Basic text cleaning
        cleaned = text.strip()
        
        # Remove excessive whitespace
        cleaned = ' '.join(cleaned.split())
        
        # MiniLM can handle longer texts, increase limit for larger chunks
        max_length = 3000  # Increased limit to accommodate larger chunks
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length]
            self.logger.warning(f"Text truncated to {max_length} characters")
        
        return cleaned
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        return 384
    
    def test_connection(self) -> bool:
        """
        Test the embedding service functionality.
        
        Returns:
            True if service is working
        """
        try:
            # Test with a simple text
            test_embedding = self.embed_text("Hello, this is a test.")
            return len(test_embedding) == self.get_embedding_dimension()
        except Exception as e:
            self.logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.get_embedding_dimension(),
            "local": True,
            "requires_api_key": False,
            "max_sequence_length": 512  # Default for MiniLM
        }
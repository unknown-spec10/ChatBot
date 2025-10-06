"""
Enhanced Vector Store for RAG System.

Implements proper cosine similarity thresholds, confidence scoring,
and industry-standard retrieval strategies.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
from ..rag.document_processor import DocumentChunk
from ..rag.minilm_embeddings import MiniLMEmbeddingService


@dataclass
class RetrievalResult:
    """Enhanced retrieval result with confidence scoring."""
    chunk: DocumentChunk
    similarity_score: float
    confidence_level: str  # "high", "medium", "low"
    rank: int


class ConfidenceEvaluator:
    """Evaluates confidence levels for RAG retrieval."""
    
    def __init__(
        self,
        high_confidence_threshold: float = 0.4,  # Adjusted for actual cosine similarity scores
        low_confidence_threshold: float = 0.25   # Adjusted for actual cosine similarity scores
    ):
        """
        Initialize confidence evaluator.
        
        Args:
            high_confidence_threshold: Threshold for high confidence (≥0.8)
            low_confidence_threshold: Threshold for low confidence (≥0.6)
        """
        self.high_threshold = high_confidence_threshold
        self.low_threshold = low_confidence_threshold
    
    def evaluate_confidence(self, max_similarity: float) -> str:
        """
        Evaluate confidence level based on maximum similarity score.
        
        Args:
            max_similarity: Highest similarity score from retrieval
            
        Returns:
            Confidence level: "high", "medium", or "low"
        """
        if max_similarity >= self.high_threshold:
            return "high"
        elif max_similarity >= self.low_threshold:
            return "medium"
        else:
            return "low"
    
    def get_rag_strategy(self, max_similarity: float) -> str:
        """
        Determine RAG strategy based on similarity scores.
        
        Args:
            max_similarity: Highest similarity score from retrieval
            
        Returns:
            Strategy: "factual", "hybrid", or "general"
        """
        if max_similarity >= self.high_threshold:
            return "factual"  # Low temperature, stick to sources
        elif max_similarity >= self.low_threshold:
            return "hybrid"   # Medium temperature, blend with general knowledge
        else:
            return "general"  # High temperature, use general knowledge


class MiniLMEmbeddingFunction:
    """ChromaDB embedding function wrapper for MiniLM."""
    
    def __init__(self, embedding_service: MiniLMEmbeddingService):
        """Initialize with MiniLM embedding service."""
        self.embedding_service = embedding_service
    
    def __call__(self, input_texts) -> List[List[float]]:
        """Generate embeddings for ChromaDB."""
        # Handle different input formats
        if hasattr(input_texts, 'input'):
            texts = input_texts.input
        elif isinstance(input_texts, dict) and 'input' in input_texts:
            texts = input_texts['input']
        else:
            texts = input_texts
        
        # Ensure we have a list of strings
        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list):
            texts = [str(texts)]
        
        # Validate all items are strings
        texts = [str(text) if not isinstance(text, str) else text for text in texts]
        
        return self.embedding_service.embed_batch(texts)


class EnhancedChromaVectorStore:
    """
    Enhanced ChromaDB vector store with confidence-based retrieval.
    
    Features:
    - Proper cosine similarity thresholds
    - Confidence scoring and evaluation
    - Enhanced metadata support
    - Industry-standard retrieval patterns
    """
    
    def __init__(
        self,
        embedding_service: MiniLMEmbeddingService,
        persist_directory: str = "./chroma_db",
        collection_name: str = "document_chunks"
    ):
        """
        Initialize the enhanced vector store.
        
        Args:
            embedding_service: MiniLM embedding service
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the collection
        """
        self.embedding_service = embedding_service
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.logger = logging.getLogger(__name__)
        
        # Initialize confidence evaluator
        self.confidence_evaluator = ConfidenceEvaluator()
        
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        self.embedding_function = MiniLMEmbeddingFunction(embedding_service)
        
        # Initialize ChromaDB client with safer settings
        try:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True  # Allow reset if needed
                )
            )
        except Exception as e:
            self.logger.warning(f"Failed to create PersistentClient with custom settings: {e}")
            # Fallback to basic client
            self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Try to initialize collection with extensive error handling
        try:
            self.collection = self._get_or_create_collection()
            self.logger.info(f"Successfully initialized ChromaDB at {persist_directory}")
        except Exception as e:
            self.logger.error(f"Failed to initialize collection: {e}")
            raise RuntimeError(f"ChromaDB initialization failed: {e}")
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one with better error handling."""
        try:
            # First try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            self.logger.info(f"Loaded existing collection: {self.collection_name}")
            return collection
        except ValueError as e:
            if "does not exist" in str(e).lower():
                # Collection doesn't exist, create it
                try:
                    collection = self.client.create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    self.logger.info(f"Created new collection: {self.collection_name}")
                    return collection
                except Exception as create_e:
                    self.logger.error(f"Failed to create collection: {create_e}")
                    raise
            else:
                # Different ValueError, try to recreate
                self.logger.warning(f"Collection access error: {e}")
                try:
                    self.client.delete_collection(name=self.collection_name)
                    collection = self.client.create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    self.logger.info(f"Recreated collection after error: {self.collection_name}")
                    return collection
                except Exception as final_e:
                    self.logger.error(f"Failed to recreate collection: {final_e}")
                    raise
        except Exception as e:
            # Any other exception
            self.logger.warning(f"Unexpected error accessing collection: {e}")
            try:
                # Try to create collection with simpler metadata
                collection = self.client.create_collection(
                    name=self.collection_name
                )
                self.logger.info(f"Created collection with default settings: {self.collection_name}")
                return collection
            except Exception as final_e:
                self.logger.error(f"Failed to create collection: {final_e}")
                raise
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of document chunks to add
        """
        try:
            # Prepare data for ChromaDB
            ids = [chunk.chunk_id for chunk in chunks]
            documents = [chunk.text for chunk in chunks]
            metadatas = []
            
            for chunk in chunks:
                metadata = {
                    "source_path": chunk.source_document,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count
                }
                
                if chunk.section_title:
                    metadata["section_title"] = chunk.section_title
                
                # Add custom metadata
                if chunk.metadata:
                    metadata.update(chunk.metadata)
                
                metadatas.append(metadata)
            
            # Generate embeddings manually
            self.logger.info(f"Generating embeddings for {len(documents)} documents...")
            embeddings = self.embedding_service.embed_batch(documents)
            
            # Convert embeddings to the format ChromaDB expects (flatten if needed)
            processed_embeddings = []
            for emb in embeddings:
                if isinstance(emb, list):
                    processed_embeddings.append(emb)
                else:
                    processed_embeddings.append(emb.tolist() if hasattr(emb, 'tolist') else list(emb))
            
            # Add to collection with manual embeddings
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=processed_embeddings
            )
            
            self.logger.info(f"Added {len(chunks)} chunks to vector store")
            
        except Exception as e:
            self.logger.error(f"Error adding chunks to vector store: {e}")
            raise
    
    def _expand_query(self, query: str) -> str:
        """
        Expand query with synonyms and related terms for better matching.
        
        Args:
            query: Original search query
            
        Returns:
            Expanded query string
        """
        # Define query expansion mappings
        expansion_mappings = {
            'vacation': ['vacation', 'PTO', 'time off', 'annual leave', 'paid time off'],
            'sick': ['sick leave', 'medical leave', 'illness', 'sick days'],
            'benefits': ['benefits', 'compensation', 'perks', 'coverage', 'insurance'],
            'remote': ['remote work', 'work from home', 'telecommuting', 'WFH'],
            'health': ['health insurance', 'medical coverage', 'healthcare', 'wellness'],
            'holiday': ['holidays', 'federal holidays', 'time off', 'observances'],
            'leave': ['leave', 'time off', 'absence', 'PTO', 'vacation']
        }
        
        query_lower = query.lower()
        expanded_terms = [query]  # Always include original
        
        for key, synonyms in expansion_mappings.items():
            if key in query_lower:
                expanded_terms.extend(synonyms)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in expanded_terms:
            if term.lower() not in seen:
                unique_terms.append(term)
                seen.add(term.lower())
        
        return ' '.join(unique_terms)

    def search(
        self,
        query: str,
        max_results: int = 5,
        include_low_confidence: bool = False
    ) -> List[RetrievalResult]:
        """
        Enhanced search with confidence-based filtering and query expansion.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            include_low_confidence: Whether to include low confidence results
            
        Returns:
            List of retrieval results with confidence scores
        """
        try:
            # Expand query for better semantic matching
            expanded_query = self._expand_query(query)
            self.logger.debug(f"Expanded query from '{query}' to '{expanded_query}'")
            
            # Generate query embedding manually to avoid ChromaDB issues
            query_embedding = self.embedding_service.embed_query(expanded_query)
            
            # Search the collection using query_embeddings instead of query_texts
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results["documents"] or not results["documents"][0]:
                return []
            
            # Convert results to RetrievalResult objects
            retrieval_results = []
            
            # Safely access results with proper null checks
            documents = results["documents"][0] if results["documents"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []
            
            for i, (doc, metadata, distance) in enumerate(zip(
                documents, metadatas, distances
            )):
                # Convert distance to cosine similarity
                # ChromaDB returns cosine distance, so similarity = 1 - distance
                similarity_score = 1 - distance
                
                # Safely extract metadata with type checking
                chunk_id = str(metadata.get("chunk_id", f"chunk_{i}"))
                source_path = str(metadata.get("source_path", "unknown"))
                section_title = metadata.get("section_title")
                section_title_str = str(section_title) if section_title else None
                
                # Safely convert chunk_index and token_count
                chunk_index_raw = metadata.get("chunk_index", i)
                chunk_index = int(chunk_index_raw) if isinstance(chunk_index_raw, (int, float, str)) and str(chunk_index_raw).isdigit() else i
                
                token_count_raw = metadata.get("token_count", 0)
                token_count = int(token_count_raw) if isinstance(token_count_raw, (int, float, str)) and str(token_count_raw).isdigit() else 0
                
                # Create document chunk from results
                chunk = DocumentChunk(
                    text=doc,
                    metadata=dict(metadata),
                    chunk_id=chunk_id,
                    source_document=source_path,
                    section_title=section_title_str,
                    chunk_index=chunk_index,
                    token_count=token_count
                )
                
                # Evaluate confidence
                confidence_level = self.confidence_evaluator.evaluate_confidence(similarity_score)
                
                # Create retrieval result
                result = RetrievalResult(
                    chunk=chunk,
                    similarity_score=similarity_score,
                    confidence_level=confidence_level,
                    rank=i + 1
                )
                
                # Filter by confidence if requested
                if include_low_confidence or confidence_level != "low":
                    retrieval_results.append(result)
            
            # Log retrieval statistics
            if retrieval_results:
                max_score = max(r.similarity_score for r in retrieval_results)
                confidence = self.confidence_evaluator.evaluate_confidence(max_score)
                self.logger.info(
                    f"Retrieved {len(retrieval_results)} chunks, "
                    f"max similarity: {max_score:.3f}, "
                    f"confidence: {confidence}"
                )
            
            return retrieval_results
            
        except Exception as e:
            self.logger.error(f"Error searching vector store: {e}")
            return []
    
    def get_rag_strategy(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Get comprehensive RAG strategy based on retrieval confidence.
        
        Args:
            query: Search query
            max_results: Maximum results to retrieve
            
        Returns:
            Dictionary with strategy information
        """
        results = self.search(query, max_results, include_low_confidence=True)
        
        if not results:
            return {
                "use_rag": False,
                "confidence": "none",
                "strategy": "general",
                "max_similarity": 0.0,
                "num_results": 0,
                "context_chunks": []
            }
        
        max_similarity = max(r.similarity_score for r in results)
        confidence = self.confidence_evaluator.evaluate_confidence(max_similarity)
        strategy = self.confidence_evaluator.get_rag_strategy(max_similarity)
        
        # Filter results based on strategy
        if strategy == "factual":
            context_chunks = [r for r in results if r.confidence_level == "high"]
        elif strategy == "hybrid":
            context_chunks = [r for r in results if r.confidence_level in ["high", "medium"]]
        else:
            context_chunks = []
        
        return {
            "use_rag": len(context_chunks) > 0,
            "confidence": confidence,
            "strategy": strategy,
            "max_similarity": max_similarity,
            "num_results": len(context_chunks),
            "context_chunks": context_chunks
        }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.
        
        Returns:
            Dictionary with store statistics
        """
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {
                "total_chunks": 0,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "error": str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Alias for get_collection_stats for compatibility."""
        return self.get_collection_stats()
    
    def clear(self) -> None:
        """Clear all data from the vector store."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self._get_or_create_collection()
            self.logger.info("Vector store cleared")
        except Exception as e:
            self.logger.error(f"Error clearing vector store: {e}")
    
    def delete_collection(self) -> None:
        """Delete the collection and recreate it (alias for clear)."""
        self.clear()
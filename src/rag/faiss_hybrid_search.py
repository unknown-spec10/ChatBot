"""
FAISS-based Hybrid Search System for Agentic RAG.

Uses FAISS for efficient vector similarity search with multiple index types
for improved recall and precision in document retrieval.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import faiss

from ..rag.document_processor import DocumentChunk
from ..rag.vector_store import RetrievalResult, EnhancedChromaVectorStore


@dataclass
class FaissHybridResult:
    """Result from FAISS hybrid search combining multiple search strategies."""
    chunk: DocumentChunk
    similarity_score: float      # Primary similarity score
    dense_score: float          # Dense vector similarity
    sparse_score: float         # Sparse/keyword-like similarity  
    combined_score: float       # Final combined score
    search_method: str          # "dense", "sparse", "hybrid"
    rank: int                   # Final ranking


class FaissHybridRetriever:
    """
    FAISS-based hybrid retriever combining dense and sparse search strategies.
    
    This replaces the BM25 approach with FAISS indices for better performance
    and easier integration with the existing vector embedding pipeline.
    """
    
    def __init__(
        self,
        vector_store: EnhancedChromaVectorStore,
        embedding_dimension: int = 384,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ):
        """
        Initialize FAISS hybrid retriever.
        
        Args:
            vector_store: ChromaDB vector store containing documents
            embedding_dimension: Dimension of embeddings (384 for MiniLM)
            dense_weight: Weight for dense similarity (0.7 = 70%)
            sparse_weight: Weight for sparse similarity (0.3 = 30%)
        """
        self.vector_store = vector_store
        self.embedding_dimension = embedding_dimension
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.logger = logging.getLogger(__name__)
        
        # FAISS indices
        self.dense_index: Optional[faiss.Index] = None
        self.sparse_index: Optional[faiss.Index] = None
        
        # Document storage
        self.documents: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        
        # Initialize indices
        self._initialize_indices()
    
    def _initialize_indices(self) -> None:
        """Initialize FAISS indices with documents from vector store."""
        try:
            self.logger.info("Initializing FAISS indices...")
            
            # Get all documents from ChromaDB
            documents = self._get_all_documents()
            
            if not documents:
                self.logger.warning("No documents found for FAISS indexing")
                return
            
            self.documents = documents
            self.logger.info(f"Found {len(documents)} documents for FAISS indexing")
            
            # Get embeddings for all documents
            embeddings = self._get_document_embeddings()
            
            if embeddings is None or len(embeddings) == 0:
                self.logger.warning("No embeddings generated for FAISS indexing")
                return
            
            self.embeddings = embeddings
            
            # Create FAISS indices
            self._create_dense_index()
            self._create_sparse_index()
            
            self.logger.info("FAISS indices initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FAISS indices: {e}")
    
    def _get_all_documents(self) -> List[DocumentChunk]:
        """Get all documents from the vector store."""
        documents = []
        
        try:
            # Method 1: Direct access to ChromaDB collection
            all_data = self.vector_store.collection.get(
                include=["documents", "metadatas"]
            )
            
            if all_data and all_data.get("documents"):
                documents_list = all_data["documents"] if all_data["documents"] is not None else []
                metadatas_list = all_data.get("metadatas", []) if all_data.get("metadatas", []) is not None else []
                # Ensure metadatas_list is a list of dicts
                if not isinstance(metadatas_list, list):
                    metadatas_list = [metadatas_list] * len(documents_list)
                for i, doc in enumerate(documents_list):
                    metadata = metadatas_list[i] if i < len(metadatas_list) else {}
                    if metadata is None:
                        metadata = {}
                    
                    chunk = DocumentChunk(
                        text=doc,
                        metadata=dict(metadata),
                        chunk_id=str(metadata.get("chunk_id", f"chunk_{i}")),
                        source_document=str(metadata.get("source_path", "unknown")),
                        section_title=str(metadata.get("section_title")) if metadata.get("section_title") else None,
                        chunk_index=int(metadata.get("chunk_index", i)),
                        token_count=int(metadata.get("token_count", 0))
                    )
                    documents.append(chunk)
                    
                self.logger.info(f"Retrieved {len(documents)} documents from ChromaDB")
                
        except Exception as e:
            self.logger.warning(f"Direct ChromaDB access failed: {e}")
            
            # Fallback: Use vector store search with common terms
            try:
                search_terms = ["the", "and", "policy", "employee", "work", "time", "benefit"]
                
                for term in search_terms:
                    results = self.vector_store.search(term, max_results=20, include_low_confidence=True)
                    for result in results:
                        if not any(d.chunk_id == result.chunk.chunk_id for d in documents):
                            documents.append(result.chunk)
                    
                    if len(documents) >= 50:
                        break
                
                self.logger.info(f"Retrieved {len(documents)} documents via search fallback")
                
            except Exception as fallback_e:
                self.logger.error(f"Fallback document retrieval failed: {fallback_e}")
        
        return documents
    
    def _get_document_embeddings(self) -> Optional[np.ndarray]:
        """Get embeddings for all documents."""
        try:
            texts = [doc.text for doc in self.documents]
            embeddings_list = self.vector_store.embedding_service.embed_batch(texts)
            
            # Convert to numpy array
            embeddings = np.array(embeddings_list, dtype=np.float32)
            
            self.logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to generate document embeddings: {e}")
            return None
    
    def _create_dense_index(self) -> None:
        """Create dense FAISS index for semantic similarity."""
        try:
            # Use IndexFlatIP for inner product (cosine similarity after normalization)
            self.dense_index = faiss.IndexFlatIP(self.embedding_dimension)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            
            # Add embeddings to index
            self.dense_index.add(self.embeddings)
            
            self.logger.info(f"Dense FAISS index created with {self.dense_index.ntotal} vectors")
            
        except Exception as e:
            self.logger.error(f"Failed to create dense FAISS index: {e}")
    
    def _create_sparse_index(self) -> None:
        """Create sparse-like index using LSH for diversity."""
        try:
            # Use LSH (Locality Sensitive Hashing) for sparse-like retrieval
            # This provides different similarity patterns than dense search
            nbits = 64  # Number of bits for LSH
            self.sparse_index = faiss.IndexLSH(self.embedding_dimension, nbits)
            
            # Add embeddings to sparse index (without normalization for different patterns)
            sparse_embeddings = self.embeddings.copy()
            self.sparse_index.add(sparse_embeddings)
            
            self.logger.info(f"Sparse FAISS index created with {self.sparse_index.ntotal} vectors")
            
        except Exception as e:
            self.logger.error(f"Failed to create sparse FAISS index: {e}")
    
    def search(
        self,
        query: str,
        max_results: int = 10,
        search_method: str = "hybrid"  # "dense", "sparse", "hybrid"
    ) -> List[FaissHybridResult]:
        """
        Perform hybrid search using FAISS indices.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            search_method: Search strategy ("dense", "sparse", "hybrid")
            
        Returns:
            List of hybrid search results
        """
        try:
            if not self.documents or self.dense_index is None:
                self.logger.warning("FAISS indices not initialized")
                return []
            
            # Generate query embedding
            query_embedding = self.vector_store.embedding_service.embed_query(query)
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            results = []
            
            if search_method in ["dense", "hybrid"]:
                dense_results = self._search_dense(query_vector, max_results * 2)
                results.extend(dense_results)
            
            if search_method in ["sparse", "hybrid"] and self.sparse_index is not None:
                sparse_results = self._search_sparse(query_vector, max_results * 2)
                results.extend(sparse_results)
            
            # Combine and rank results
            final_results = self._combine_results(results, max_results, search_method)
            
            self.logger.info(f"FAISS hybrid search returned {len(final_results)} results")
            return final_results
            
        except Exception as e:
            self.logger.error(f"FAISS hybrid search failed: {e}")
            return []
    
    def _search_dense(self, query_vector: np.ndarray, k: int) -> List[FaissHybridResult]:
        """Search using dense FAISS index."""
        try:
            # Normalize query vector for cosine similarity
            faiss.normalize_L2(query_vector)
            
            # Search dense index
            scores, indices = self.dense_index.search(query_vector, min(k, self.dense_index.ntotal))
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= 0 and idx < len(self.documents):  # Valid index
                    result = FaissHybridResult(
                        chunk=self.documents[idx],
                        similarity_score=float(score),
                        dense_score=float(score),
                        sparse_score=0.0,
                        combined_score=float(score) * self.dense_weight,
                        search_method="dense",
                        rank=i + 1
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Dense search failed: {e}")
            return []
    
    def _search_sparse(self, query_vector: np.ndarray, k: int) -> List[FaissHybridResult]:
        """Search using sparse FAISS index."""
        try:
            # Search sparse index (no normalization for different patterns)
            scores, indices = self.sparse_index.search(query_vector, min(k, self.sparse_index.ntotal))
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= 0 and idx < len(self.documents):  # Valid index
                    # Convert LSH distance to similarity (invert and normalize)
                    similarity = 1.0 / (1.0 + abs(float(score)))
                    
                    result = FaissHybridResult(
                        chunk=self.documents[idx],
                        similarity_score=similarity,
                        dense_score=0.0,
                        sparse_score=similarity,
                        combined_score=similarity * self.sparse_weight,
                        search_method="sparse",
                        rank=i + 1
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Sparse search failed: {e}")
            return []
    
    def _combine_results(
        self,
        results: List[FaissHybridResult],
        max_results: int,
        search_method: str
    ) -> List[FaissHybridResult]:
        """Combine and rank results from different search methods."""
        try:
            # Remove duplicates and combine scores
            combined = {}
            
            for result in results:
                chunk_id = result.chunk.chunk_id
                
                if chunk_id in combined:
                    # Combine scores
                    existing = combined[chunk_id]
                    existing.dense_score = max(existing.dense_score, result.dense_score)
                    existing.sparse_score = max(existing.sparse_score, result.sparse_score)
                    existing.combined_score = (
                        existing.dense_score * self.dense_weight +
                        existing.sparse_score * self.sparse_weight
                    )
                    existing.similarity_score = existing.combined_score
                    existing.search_method = "hybrid" if search_method == "hybrid" else result.search_method
                else:
                    combined[chunk_id] = result
            
            # Sort by combined score
            final_results = list(combined.values())
            final_results.sort(key=lambda x: x.combined_score, reverse=True)
            
            # Update ranks and limit results
            for i, result in enumerate(final_results[:max_results]):
                result.rank = i + 1
            
            return final_results[:max_results]
            
        except Exception as e:
            self.logger.error(f"Failed to combine results: {e}")
            return results[:max_results]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get FAISS retriever statistics."""
        return {
            "total_documents": len(self.documents),
            "embedding_dimension": self.embedding_dimension,
            "dense_index_size": self.dense_index.ntotal if self.dense_index else 0,
            "sparse_index_size": self.sparse_index.ntotal if self.sparse_index else 0,
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
            "indices_initialized": self.dense_index is not None
        }


def convert_faiss_to_retrieval_results(
    faiss_results: List[FaissHybridResult]
) -> List[RetrievalResult]:
    """Convert FAISS results to standard RetrievalResult format."""
    retrieval_results = []
    
    for faiss_result in faiss_results:
        # Determine confidence level based on similarity score
        if faiss_result.similarity_score >= 0.4:
            confidence = "high"
        elif faiss_result.similarity_score >= 0.2:
            confidence = "medium"
        else:
            confidence = "low"
        
        result = RetrievalResult(
            chunk=faiss_result.chunk,
            similarity_score=faiss_result.similarity_score,
            confidence_level=confidence,
            rank=faiss_result.rank
        )
        retrieval_results.append(result)
    
    return retrieval_results
"""
Cross-Encoder Re-ranker for Agentic RAG.

Implements sophisticated document re-ranking using cross-encoder models
to refine the top 50 retrieved chunks and select the definitive top 5.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    CrossEncoder = None

from ..rag.document_processor import DocumentChunk
from ..rag.faiss_hybrid_search import FaissHybridResult


@dataclass
class RerankedResult:
    """Result from cross-encoder re-ranking."""
    chunk: DocumentChunk
    original_score: float       # Original hybrid score
    cross_encoder_score: float  # Cross-encoder relevance score
    rerank_position: int        # New position after re-ranking
    confidence_level: str       # "very_high", "high", "medium", "low"
    relevance_explanation: str  # Why this chunk is relevant


class CrossEncoderReranker:
    """
    Cross-encoder re-ranker for precise relevance scoring.
    
    Uses a specialized cross-encoder model to score query-document pairs
    for more accurate relevance assessment than bi-encoder approaches.
    """
    
    def __init__(
        self, 
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 16,
        use_fallback: bool = True
    ):
        """
        Initialize cross-encoder re-ranker.
        
        Args:
            model_name: HuggingFace model name for cross-encoder
            batch_size: Batch size for processing
            use_fallback: Whether to use fallback when model unavailable
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_fallback = use_fallback
        self.model = None
        
        self.logger = logging.getLogger(__name__)
        
        # Load cross-encoder model
        self._load_model()
        
        # Confidence thresholds for cross-encoder scores
        self.confidence_thresholds = {
            "very_high": 0.8,
            "high": 0.6,
            "medium": 0.4,
            "low": 0.0
        }
    
    def _load_model(self) -> None:
        """Load the cross-encoder model."""
        if not CROSS_ENCODER_AVAILABLE:
            if self.use_fallback:
                self.logger.warning(
                    "sentence-transformers not available. Using fallback re-ranking."
                )
                return
            else:
                raise ImportError(
                    "sentence-transformers required for cross-encoder re-ranking. "
                    "Install with: pip install sentence-transformers"
                )
        
        try:
            self.logger.info(f"Loading cross-encoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            self.logger.info("Cross-encoder model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load cross-encoder model: {e}")
            if not self.use_fallback:
                raise
            self.model = None
    
    def rerank(
        self, 
        query: str, 
        hybrid_results: List[FaissHybridResult],
        top_k: int = 5,
        rerank_top_n: int = 50
    ) -> List[RerankedResult]:
        """
        Re-rank hybrid results using cross-encoder model.
        
        Args:
            query: Original search query
            hybrid_results: Results from hybrid search
            top_k: Number of top results to return
            rerank_top_n: Number of top results to re-rank
            
        Returns:
            List of re-ranked results with cross-encoder scores
        """
        if not hybrid_results:
            return []
        
        self.logger.info(
            f"Re-ranking top {min(rerank_top_n, len(hybrid_results))} results "
            f"to select top {top_k}"
        )
        
        # Take top N results for re-ranking
        candidates = hybrid_results[:rerank_top_n]
        
        if self.model is not None:
            # Use cross-encoder model
            reranked = self._rerank_with_model(query, candidates, top_k)
        else:
            # Use fallback re-ranking
            reranked = self._rerank_fallback(query, candidates, top_k)
        
        self.logger.info(
            f"Re-ranking complete: {len(reranked)} results returned"
        )
        
        return reranked
    
    def _rerank_with_model(
        self, 
        query: str, 
        candidates: List[FaissHybridResult],
        top_k: int
    ) -> List[RerankedResult]:
        """Re-rank using cross-encoder model."""
        
        # Prepare query-document pairs
        query_doc_pairs = []
        for candidate in candidates:
            # Use first 512 characters of chunk text to avoid token limits
            doc_text = candidate.chunk.text[:512]
            query_doc_pairs.append([query, doc_text])
        
        try:
            # Get cross-encoder scores in batches
            all_scores = []
            
            for i in range(0, len(query_doc_pairs), self.batch_size):
                batch = query_doc_pairs[i:i + self.batch_size]
                batch_scores = self.model.predict(batch)
                all_scores.extend(batch_scores)
            
            # Convert to numpy array for easier handling
            scores = np.array(all_scores)
            
            # Create scored results
            scored_results = []
            for i, (candidate, score) in enumerate(zip(candidates, scores)):
                scored_results.append((candidate, float(score), i))
            
            # Sort by cross-encoder score (descending)
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # Create reranked results
            reranked_results = []
            for new_pos, (candidate, ce_score, original_pos) in enumerate(scored_results[:top_k]):
                
                confidence = self._get_confidence_level(ce_score)
                explanation = self._generate_relevance_explanation(
                    query, candidate.chunk.text, ce_score
                )
                
                reranked_result = RerankedResult(
                    chunk=candidate.chunk,
                    original_score=candidate.combined_score,
                    cross_encoder_score=ce_score,
                    rerank_position=new_pos + 1,
                    confidence_level=confidence,
                    relevance_explanation=explanation
                )
                
                reranked_results.append(reranked_result)
            
            return reranked_results
            
        except Exception as e:
            self.logger.error(f"Cross-encoder re-ranking failed: {e}")
            # Fallback to original ranking
            return self._rerank_fallback(query, candidates, top_k)
    
    def _rerank_fallback(
        self, 
        query: str, 
        candidates: List[FaissHybridResult],
        top_k: int
    ) -> List[RerankedResult]:
        """Fallback re-ranking using heuristic methods."""
        self.logger.info("Using fallback re-ranking (no cross-encoder model)")
        
        # Simple heuristic re-ranking based on:
        # 1. Original hybrid score
        # 2. Query term overlap
        # 3. Chunk length penalty
        # 4. Source document diversity
        
        query_terms = set(query.lower().split())
        
        enhanced_candidates = []
        for candidate in candidates:
            # Calculate query term overlap
            chunk_terms = set(candidate.chunk.text.lower().split())
            overlap_ratio = len(query_terms & chunk_terms) / len(query_terms) if query_terms else 0
            
            # Length penalty (very short or very long chunks are penalized)
            ideal_length = 200  # characters
            length_penalty = abs(len(candidate.chunk.text) - ideal_length) / ideal_length
            length_penalty = min(length_penalty, 1.0)
            
            # Combine scores
            fallback_score = (
                0.5 * candidate.combined_score +      # Original hybrid score
                0.3 * overlap_ratio +              # Query term overlap
                0.2 * (1.0 - length_penalty)       # Length penalty (inverted)
            )
            
            enhanced_candidates.append((candidate, fallback_score))
        
        # Sort by enhanced score
        enhanced_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Create reranked results
        reranked_results = []
        for new_pos, (candidate, fallback_score) in enumerate(enhanced_candidates[:top_k]):
            
            confidence = self._get_confidence_level(fallback_score)
            explanation = f"Heuristic re-ranking based on hybrid score and query overlap"
            
            reranked_result = RerankedResult(
                chunk=candidate.chunk,
                original_score=candidate.combined_score,
                cross_encoder_score=fallback_score,
                rerank_position=new_pos + 1,
                confidence_level=confidence,
                relevance_explanation=explanation
            )
            
            reranked_results.append(reranked_result)
        
        return reranked_results
    
    def _get_confidence_level(self, score: float) -> str:
        """Get confidence level based on cross-encoder score."""
        if score >= self.confidence_thresholds["very_high"]:
            return "very_high"
        elif score >= self.confidence_thresholds["high"]:
            return "high"
        elif score >= self.confidence_thresholds["medium"]:
            return "medium"
        else:
            return "low"
    
    def _generate_relevance_explanation(
        self, query: str, chunk_text: str, score: float
    ) -> str:
        """Generate explanation for why a chunk is relevant."""
        
        confidence = self._get_confidence_level(score)
        
        if confidence == "very_high":
            return f"Highly relevant match (score: {score:.3f}) - strong semantic alignment with query"
        elif confidence == "high":
            return f"Good relevance (score: {score:.3f}) - clear connection to query topic"
        elif confidence == "medium":
            return f"Moderate relevance (score: {score:.3f}) - partial match to query"
        else:
            return f"Low relevance (score: {score:.3f}) - weak connection to query"
    
    def batch_rerank(
        self, 
        queries: List[str], 
        results_list: List[List[FaissHybridResult]],
        top_k: int = 5
    ) -> List[List[RerankedResult]]:
        """
        Batch re-rank multiple query-results pairs.
        
        Args:
            queries: List of queries
            results_list: List of hybrid results for each query
            top_k: Number of top results to return for each query
            
        Returns:
            List of re-ranked results for each query
        """
        self.logger.info(f"Batch re-ranking {len(queries)} queries")
        
        reranked_all = []
        for query, results in zip(queries, results_list):
            reranked = self.rerank(query, results, top_k)
            reranked_all.append(reranked)
        
        return reranked_all
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the cross-encoder model."""
        return {
            "model_name": self.model_name,
            "model_loaded": self.model is not None,
            "cross_encoder_available": CROSS_ENCODER_AVAILABLE,
            "batch_size": self.batch_size,
            "confidence_thresholds": self.confidence_thresholds,
            "fallback_enabled": self.use_fallback
        }


class AdaptiveReranker:
    """
    Adaptive re-ranker that chooses the best re-ranking strategy
    based on query characteristics and available models.
    """
    
    def __init__(self):
        """Initialize adaptive re-ranker."""
        self.cross_encoder = CrossEncoderReranker()
        self.logger = logging.getLogger(__name__)
    
    def rerank(
        self, 
        query: str, 
        hybrid_results: List[FaissHybridResult],
        top_k: int = 5,
        query_type: Optional[str] = None
    ) -> List[RerankedResult]:
        """
        Adaptively re-rank results based on query characteristics.
        
        Args:
            query: Search query
            hybrid_results: Hybrid search results
            top_k: Number of results to return
            query_type: Optional query type hint
            
        Returns:
            Re-ranked results
        """
        if not hybrid_results:
            return []
        
        # Analyze query characteristics
        query_analysis = self._analyze_query(query, query_type)
        
        # Choose re-ranking strategy
        if query_analysis["complexity"] == "high" and len(hybrid_results) > 10:
            # Use more aggressive re-ranking for complex queries
            rerank_top_n = min(50, len(hybrid_results))
        elif query_analysis["specificity"] == "high":
            # Focus on top candidates for specific queries
            rerank_top_n = min(20, len(hybrid_results))
        else:
            # Standard re-ranking
            rerank_top_n = min(30, len(hybrid_results))
        
        self.logger.info(
            f"Adaptive re-ranking: complexity={query_analysis['complexity']}, "
            f"specificity={query_analysis['specificity']}, "
            f"reranking top {rerank_top_n} candidates"
        )
        
        return self.cross_encoder.rerank(
            query, hybrid_results, top_k, rerank_top_n
        )
    
    def _analyze_query(self, query: str, query_type: Optional[str] = None) -> Dict[str, str]:
        """Analyze query characteristics for adaptive re-ranking."""
        
        words = query.split()
        
        # Complexity analysis
        if len(words) > 10 or "and" in query.lower() or "or" in query.lower():
            complexity = "high"
        elif len(words) > 5:
            complexity = "medium"
        else:
            complexity = "low"
        
        # Specificity analysis
        specific_terms = ["policy", "procedure", "form", "contact", "phone", "email", "address"]
        if any(term in query.lower() for term in specific_terms):
            specificity = "high"
        elif any(word.isupper() for word in words):  # Contains acronyms/proper nouns
            specificity = "high"
        else:
            specificity = "medium"
        
        return {
            "complexity": complexity,
            "specificity": specificity,
            "length": "long" if len(words) > 8 else "short",
            "type": query_type or "unknown"
        }
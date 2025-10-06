"""
Agentic RAG System with Plan → Act → Reflect Cycles.

This module implements a sophisticated multi-step RAG system that uses
an agent-based approach for intelligent query processing and retrieval.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

from ..models.message import Message, MessageRole
from ..interfaces.core import IAIProvider
from ..rag.vector_store import EnhancedChromaVectorStore, RetrievalResult


class AgentThought(Enum):
    """Types of agent thoughts in the Plan → Act → Reflect cycle."""
    QUERY_ANALYSIS = "query_analysis"
    INTENT_ROUTING = "intent_routing"
    SEARCH_EXECUTION = "search_execution"
    CONFIDENCE_ASSESSMENT = "confidence_assessment"
    REFLECTION = "reflection"
    FINAL_GENERATION = "final_generation"


@dataclass
class AgentStep:
    """Represents a single step in the agent's reasoning process."""
    thought_type: AgentThought
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence_score: float
    reasoning: str
    timestamp: float


@dataclass
class QueryDecomposition:
    """Result of query decomposition process."""
    original_query: str
    sub_queries: List[str]
    intent_type: str  # "factual", "procedural", "complex", "conversational"
    complexity_score: float
    requires_multi_step: bool


@dataclass
class ConfidenceAssessment:
    """LLM-as-a-Judge confidence assessment."""
    context_quality_score: float  # 1.0 to 5.0
    query_coverage_score: float   # 1.0 to 5.0
    overall_confidence: float     # 1.0 to 5.0
    should_reflect: bool
    reasoning: str


class QueryProcessor:
    """Handles query decomposition and rewriting."""
    
    def __init__(self, ai_provider: IAIProvider):
        """Initialize query processor."""
        self.ai_provider = ai_provider
        self.logger = logging.getLogger(__name__)
    
    async def decompose_query(self, query: str) -> QueryDecomposition:
        """
        Decompose complex queries into simpler sub-queries.
        
        Args:
            query: Original user query
            
        Returns:
            QueryDecomposition with analysis results
        """
        decomposition_prompt = f"""
Analyze this user query and decompose it if necessary:

QUERY: "{query}"

Your task:
1. Determine if this is a simple or complex query
2. If complex, break it into 2-3 simpler sub-queries
3. Classify the intent type
4. Rate complexity (1.0 = simple, 5.0 = very complex)

Respond in JSON format:
{{
    "original_query": "{query}",
    "sub_queries": ["sub-query 1", "sub-query 2"],
    "intent_type": "factual|procedural|complex|conversational",
    "complexity_score": 3.5,
    "requires_multi_step": true,
    "reasoning": "Explanation of analysis"
}}

If the query is simple, sub_queries should contain just the original query.
"""
        
        try:
            messages = [Message(
                role=MessageRole.USER,
                content=decomposition_prompt,
                timestamp=None
            )]
            
            response = await self.ai_provider.generate_response(
                messages
            )
            
            # Parse JSON response
            try:
                result = json.loads(response.strip())
                return QueryDecomposition(
                    original_query=result["original_query"],
                    sub_queries=result["sub_queries"],
                    intent_type=result["intent_type"],
                    complexity_score=result["complexity_score"],
                    requires_multi_step=result["requires_multi_step"]
                )
            except json.JSONDecodeError:
                # Fallback for non-JSON responses
                return QueryDecomposition(
                    original_query=query,
                    sub_queries=[query],
                    intent_type="conversational",
                    complexity_score=1.0,
                    requires_multi_step=False
                )
        
        except Exception as e:
            self.logger.error(f"Query decomposition failed: {e}")
            return QueryDecomposition(
                original_query=query,
                sub_queries=[query],
                intent_type="conversational",
                complexity_score=1.0,
                requires_multi_step=False
            )


class ConfidenceJudge:
    """LLM-as-a-Judge for assessing retrieval quality."""
    
    def __init__(self, ai_provider: IAIProvider):
        """Initialize confidence judge."""
        self.ai_provider = ai_provider
        self.logger = logging.getLogger(__name__)
    
    async def assess_confidence(
        self, 
        query: str, 
        retrieved_context: str,
        num_chunks: int
    ) -> ConfidenceAssessment:
        """
        Assess the quality of retrieved context using LLM-as-a-Judge.
        
        Args:
            query: Original user query
            retrieved_context: Retrieved document context
            num_chunks: Number of chunks retrieved
            
        Returns:
            ConfidenceAssessment with detailed scores
        """
        judge_prompt = f"""
You are an expert judge evaluating the quality of retrieved documents for a user query.

QUERY: "{query}"

RETRIEVED CONTEXT:
{retrieved_context}

RETRIEVED CHUNKS: {num_chunks}

Rate the following on a scale of 1.0 to 5.0:

1. CONTEXT QUALITY: How well do the retrieved documents contain relevant information?
   - 5.0: Perfect match, comprehensive coverage
   - 4.0: Very good, covers most aspects
   - 3.0: Good, covers some aspects
   - 2.0: Poor, minimal relevant info
   - 1.0: No relevant information

2. QUERY COVERAGE: How well does the context answer the specific question?
   - 5.0: Completely answers the question
   - 4.0: Answers most of the question
   - 3.0: Partially answers the question
   - 2.0: Barely addresses the question
   - 1.0: Doesn't address the question

Respond in JSON format:
{{
    "context_quality_score": 4.2,
    "query_coverage_score": 3.8,
    "overall_confidence": 4.0,
    "should_reflect": false,
    "reasoning": "Detailed explanation of scores"
}}

Set "should_reflect" to true if overall_confidence < 3.0
"""
        
        try:
            messages = [Message(
                role=MessageRole.USER,
                content=judge_prompt,
                timestamp=None
            )]
            
            response = await self.ai_provider.generate_response(
                messages
            )
            
            # Parse JSON response
            try:
                result = json.loads(response.strip())
                return ConfidenceAssessment(
                    context_quality_score=result["context_quality_score"],
                    query_coverage_score=result["query_coverage_score"],
                    overall_confidence=result["overall_confidence"],
                    should_reflect=result["should_reflect"],
                    reasoning=result["reasoning"]
                )
            except json.JSONDecodeError:
                # Fallback assessment
                return ConfidenceAssessment(
                    context_quality_score=2.5,
                    query_coverage_score=2.5,
                    overall_confidence=2.5,
                    should_reflect=True,
                    reasoning="Failed to parse assessment"
                )
        
        except Exception as e:
            self.logger.error(f"Confidence assessment failed: {e}")
            return ConfidenceAssessment(
                context_quality_score=1.0,
                query_coverage_score=1.0,
                overall_confidence=1.0,
                should_reflect=True,
                reasoning=f"Assessment error: {e}"
            )


class AgenticRAGAgent:
    """
    Main Agentic RAG Agent implementing Plan → Act → Reflect cycles.
    
    This agent orchestrates the multi-step RAG process:
    1. Query Analysis & Planning
    2. Intent Routing
    3. Intelligent Retrieval
    4. Confidence Assessment
    5. Reflection & Adaptation
    6. Final Generation
    """
    
    def __init__(
        self,
        ai_provider: IAIProvider,
        vector_store: EnhancedChromaVectorStore,
        hybrid_retriever=None,
        max_reflection_cycles: int = 2
    ):
        """
        Initialize the Agentic RAG Agent.
        
        Args:
            ai_provider: AI service provider
            vector_store: Enhanced vector store for retrieval
            hybrid_retriever: FAISS hybrid retriever for enhanced search
            max_reflection_cycles: Maximum number of reflection cycles
        """
        self.ai_provider = ai_provider
        self.vector_store = vector_store
        self.hybrid_retriever = hybrid_retriever
        self.max_reflection_cycles = max_reflection_cycles
        
        self.query_processor = QueryProcessor(ai_provider)
        self.confidence_judge = ConfidenceJudge(ai_provider)
        
        self.logger = logging.getLogger(__name__)
        
        # Track agent steps for debugging and analysis
        self.agent_steps: List[AgentStep] = []
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query using the full Agentic RAG pipeline.
        
        Args:
            query: User query to process
            
        Returns:
            Dictionary with response and metadata
        """
        import time
        start_time = time.time()
        
        self.logger.info(f"[AGENTIC] Starting Agentic RAG processing for: {query[:50]}...")
        self.agent_steps.clear()
        
        # Stage I: Analysis & Planning
        decomposition = await self._plan_query_analysis(query)
        
        if decomposition.requires_multi_step:
            return await self._process_complex_query(decomposition)
        else:
            return await self._process_simple_query(decomposition)
    
    async def _plan_query_analysis(self, query: str) -> QueryDecomposition:
        """Stage I.1: Query Pre-Processing and Decomposition."""
        self.logger.info("[PLAN] Stage I.1: Query Analysis & Planning")
        
        import time
        step_start = time.time()
        
        decomposition = await self.query_processor.decompose_query(query)
        
        # Record agent step
        step = AgentStep(
            thought_type=AgentThought.QUERY_ANALYSIS,
            input_data={"query": query},
            output_data={
                "sub_queries": decomposition.sub_queries,
                "intent_type": decomposition.intent_type,
                "complexity_score": decomposition.complexity_score
            },
            confidence_score=min(decomposition.complexity_score / 5.0, 1.0),
            reasoning=f"Decomposed query into {len(decomposition.sub_queries)} parts",
            timestamp=step_start
        )
        self.agent_steps.append(step)
        
        self.logger.info(
            f"[DECOMPOSE] Query decomposition: {len(decomposition.sub_queries)} sub-queries, "
            f"intent: {decomposition.intent_type}, "
            f"complexity: {decomposition.complexity_score:.1f}"
        )
        
        return decomposition
    
    async def _process_simple_query(self, decomposition: QueryDecomposition) -> Dict[str, Any]:
        """Process simple queries with standard retrieval."""
        query = decomposition.original_query
        
        # Stage II: Intelligent Retrieval
        retrieval_results = await self._execute_retrieval(query)
        
        # Stage III: Confidence Assessment
        assessment = await self._assess_confidence(query, retrieval_results)
        
        if assessment.should_reflect and len(self.agent_steps) < self.max_reflection_cycles:
            self.logger.info("[REFLECT] Low confidence detected, entering reflection cycle")
            return await self._reflect_and_retry(query, assessment)
        
        # Stage IV: Final Generation
        return await self._generate_final_response(query, retrieval_results, assessment)
    
    async def _process_complex_query(self, decomposition: QueryDecomposition) -> Dict[str, Any]:
        """Process complex queries with multi-step retrieval."""
        all_results = []
        
        # Process each sub-query
        for i, sub_query in enumerate(decomposition.sub_queries):
            self.logger.info(f"[SEARCH] Processing sub-query {i+1}/{len(decomposition.sub_queries)}: {sub_query}")
            
            sub_results = await self._execute_retrieval(sub_query)
            all_results.extend(sub_results)
        
        # Combine and assess overall results
        combined_context = self._combine_retrieval_results(all_results)
        assessment = await self._assess_confidence(decomposition.original_query, all_results)
        
        if assessment.should_reflect and len(self.agent_steps) < self.max_reflection_cycles:
            self.logger.info("[REFLECT] Complex query reflection cycle")
            return await self._reflect_and_retry(decomposition.original_query, assessment)
        
        return await self._generate_final_response(
            decomposition.original_query, all_results, assessment
        )
    
    async def _execute_retrieval(self, query: str) -> List[RetrievalResult]:
        """Stage II: Execute intelligent retrieval."""
        import time
        step_start = time.time()
        
        self.logger.info("[RETRIEVE] Stage II: Intelligent Retrieval")
        
        # Initialize default values
        results = []
        max_similarity = 0.0
        confidence = "low"
        rag_strategy = {"max_similarity": 0.0, "confidence": "low", "results": []}
        
        # Use FAISS hybrid retriever if available, otherwise fallback to vector store
        self.logger.info(f"[DEBUG] Hybrid retriever available: {self.hybrid_retriever is not None}")
        if self.hybrid_retriever:
            self.logger.info("[DEBUG] Using FAISS hybrid search")
            # Use FAISS hybrid search
            faiss_results = self.hybrid_retriever.search(query, max_results=10, search_method="hybrid")
            
            # Convert FAISS results to RetrievalResult format
            from ..rag.faiss_hybrid_search import convert_faiss_to_retrieval_results
            results = convert_faiss_to_retrieval_results(faiss_results)
            
            max_similarity = max(r.similarity_score for r in results) if results else 0.0
            confidence = "high" if max_similarity > 0.4 else ("medium" if max_similarity > 0.2 else "low")
            
            # Update rag_strategy with results
            rag_strategy = {
                "max_similarity": max_similarity,
                "confidence": confidence,
                "results": results
            }
            
            self.logger.info(f"[DEBUG] FAISS search: {len(results)} results, max_similarity: {max_similarity}, confidence: {confidence}")
            
        else:
            self.logger.info("[DEBUG] Using fallback vector store search")
            # Fallback to old vector store method
            rag_strategy = self.vector_store.get_rag_strategy(query, max_results=10)
            results = rag_strategy.get("results", [])
            max_similarity = rag_strategy.get("max_similarity", 0.0)
            confidence = rag_strategy.get("confidence", "low")
        
        # Record agent step
        step = AgentStep(
            thought_type=AgentThought.SEARCH_EXECUTION,
            input_data={"query": query},
            output_data={
                "num_results": len(results),
                "max_similarity": max_similarity,
                "confidence": confidence
            },
            confidence_score=max_similarity,
            reasoning=f"Retrieved {len(results)} chunks with {confidence} confidence",
            timestamp=step_start
        )
        self.agent_steps.append(step)
        
        self.logger.info(
            f"[RESULTS] Retrieved {len(results)} chunks, "
            f"max similarity: {rag_strategy.get('max_similarity', 0.0):.3f}"
        )
        
        return results
    
    async def _assess_confidence(
        self, query: str, results: List[RetrievalResult]
    ) -> ConfidenceAssessment:
        """Stage III: Confidence Assessment using LLM-as-a-Judge."""
        import time
        step_start = time.time()
        
        self.logger.info("[ASSESS] Stage III: Confidence Assessment")
        
        if not results:
            return ConfidenceAssessment(
                context_quality_score=1.0,
                query_coverage_score=1.0,
                overall_confidence=1.0,
                should_reflect=True,
                reasoning="No results retrieved"
            )
        
        # Combine context from top results
        context_parts = []
        for result in results[:5]:  # Use top 5 for assessment
            if result.confidence_level in ["high", "medium"]:
                context_parts.append(result.chunk.text)
        
        combined_context = "\n\n---\n\n".join(context_parts)
        
        # Get LLM-as-a-Judge assessment
        assessment = await self.confidence_judge.assess_confidence(
            query, combined_context, len(results)
        )
        
        # Record agent step
        step = AgentStep(
            thought_type=AgentThought.CONFIDENCE_ASSESSMENT,
            input_data={"query": query, "num_chunks": len(results)},
            output_data={
                "context_quality": assessment.context_quality_score,
                "query_coverage": assessment.query_coverage_score,
                "overall_confidence": assessment.overall_confidence,
                "should_reflect": assessment.should_reflect
            },
            confidence_score=assessment.overall_confidence / 5.0,
            reasoning=assessment.reasoning,
            timestamp=step_start
        )
        self.agent_steps.append(step)
        
        self.logger.info(
            f"📈 Confidence scores - Quality: {assessment.context_quality_score:.1f}, "
            f"Coverage: {assessment.query_coverage_score:.1f}, "
            f"Overall: {assessment.overall_confidence:.1f}"
        )
        
        return assessment
    
    async def _reflect_and_retry(
        self, query: str, assessment: ConfidenceAssessment
    ) -> Dict[str, Any]:
        """Stage III.5: Reflection and query refinement."""
        import time
        step_start = time.time()
        
        self.logger.info("🤔 Stage III.5: Reflection & Adaptation")
        
        # Generate refined query based on confidence assessment
        refinement_prompt = f"""
The initial query retrieval had low confidence. Help refine the query for better results.

ORIGINAL QUERY: "{query}"
CONFIDENCE ISSUE: {assessment.reasoning}

Generate 2-3 alternative query formulations that might retrieve better information:
1. A more specific version
2. A broader version
3. A differently worded version

Respond with just the alternative queries, one per line.
"""
        
        try:
            messages = [Message(
                role=MessageRole.USER,
                content=refinement_prompt,
                timestamp=None
            )]
            
            response = await self.ai_provider.generate_response(
                messages
            )
            
            refined_queries = [q.strip() for q in response.split('\n') if q.strip()]
            
            # Try the first refined query
            if refined_queries:
                refined_query = refined_queries[0]
                self.logger.info(f"[REFINE] Trying refined query: {refined_query}")
                
                # Recursive call with refined query
                refined_results = await self._execute_retrieval(refined_query)
                refined_assessment = await self._assess_confidence(refined_query, refined_results)
                
                # Record reflection step
                step = AgentStep(
                    thought_type=AgentThought.REFLECTION,
                    input_data={"original_query": query, "refined_query": refined_query},
                    output_data={
                        "refined_confidence": refined_assessment.overall_confidence,
                        "improvement": refined_assessment.overall_confidence - assessment.overall_confidence
                    },
                    confidence_score=refined_assessment.overall_confidence / 5.0,
                    reasoning=f"Refined query improved confidence by {refined_assessment.overall_confidence - assessment.overall_confidence:.1f}",
                    timestamp=step_start
                )
                self.agent_steps.append(step)
                
                return await self._generate_final_response(query, refined_results, refined_assessment)
        
        except Exception as e:
            self.logger.error(f"Reflection failed: {e}")
        
        # Fallback to original results
        return await self._generate_final_response(query, [], assessment)
    
    async def _generate_final_response(
        self, 
        query: str, 
        results: List[RetrievalResult], 
        assessment: ConfidenceAssessment
    ) -> Dict[str, Any]:
        """Stage IV: Final Generation with source attribution."""
        import time
        step_start = time.time()
        
        self.logger.info("[GENERATE] Stage IV: Final Generation")
        
        if not results or assessment.overall_confidence < 2.0:
            # Low confidence - use general knowledge
            response = await self._generate_general_response(query)
            generation_strategy = "general_knowledge"
            temperature = 0.8
        elif assessment.overall_confidence >= 4.0:
            # High confidence - factual response
            response = await self._generate_factual_response(query, results)
            generation_strategy = "factual"
            temperature = 0.1
        else:
            # Medium confidence - hybrid response
            response = await self._generate_hybrid_response(query, results)
            generation_strategy = "hybrid"
            temperature = 0.4
        
        # Record final generation step
        step = AgentStep(
            thought_type=AgentThought.FINAL_GENERATION,
            input_data={"query": query, "strategy": generation_strategy},
            output_data={
                "response_length": len(response),
                "temperature": temperature,
                "num_sources": len(results)
            },
            confidence_score=assessment.overall_confidence / 5.0,
            reasoning=f"Generated {generation_strategy} response using temperature {temperature}",
            timestamp=step_start
        )
        self.agent_steps.append(step)
        
        # Compile metadata
        metadata = {
            "agent_steps": len(self.agent_steps),
            "retrieval_results": len(results),
            "confidence_assessment": {
                "context_quality": assessment.context_quality_score,
                "query_coverage": assessment.query_coverage_score,
                "overall_confidence": assessment.overall_confidence
            },
            "generation_strategy": generation_strategy,
            "temperature_used": temperature,
            "sources": [
                {
                    "document": result.chunk.source_document,
                    "similarity": result.similarity_score,
                    "confidence": result.confidence_level
                }
                for result in results[:5]  # Top 5 sources
            ]
        }
        
        self.logger.info(
            f"[SUCCESS] Agentic RAG complete: {len(self.agent_steps)} steps, "
            f"{generation_strategy} strategy, confidence {assessment.overall_confidence:.1f}"
        )
        
        return {
            "response": response,
            "metadata": metadata,
            "agent_reasoning": self.agent_steps
        }
    
    async def _generate_factual_response(self, query: str, results: List[RetrievalResult]) -> str:
        """Generate factual response with strict source attribution."""
        # Build context with source attribution
        context_parts = []
        for i, result in enumerate(results[:5]):
            if result.confidence_level in ["high", "medium"]:
                source_name = result.chunk.source_document.split('/')[-1].split('\\')[-1]
                context_parts.append(
                    f"[Source {i+1}: {source_name}]\n{result.chunk.text}"
                )
        
        context = "\n\n---\n\n".join(context_parts)
        
        factual_prompt = f"""
You are an expert assistant providing comprehensive information from company documents.

COMPANY DOCUMENT SOURCES:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Provide a COMPLETE and COMPREHENSIVE answer using ALL relevant information from the sources
- Include source attribution for every claim using [Source 1], [Source 2] format
- If multiple sources cover different aspects, synthesize them into a complete answer
- Include specific details like numbers, dates, policies, and procedures when available
- Structure your response clearly with headings or bullet points when appropriate
- If any aspect of the question isn't covered, explicitly state "The provided documents do not contain information about [specific aspect]"
- Be thorough and detailed rather than brief

RESPONSE FORMAT:
**[Main Topic]**

[Comprehensive answer with source citations covering all available information]

Sources used: [List the source documents referenced]

RESPONSE:
"""
        
        messages = [Message(role=MessageRole.USER, content=factual_prompt, timestamp=None)]
        return await self.ai_provider.generate_response(messages)
    
    async def _generate_hybrid_response(self, query: str, results: List[RetrievalResult]) -> str:
        """Generate hybrid response blending sources with general knowledge."""
        context_parts = []
        for result in results[:3]:
            if result.confidence_level in ["high", "medium"]:
                context_parts.append(result.chunk.text)
        
        context = "\n\n---\n\n".join(context_parts)
        
        hybrid_prompt = f"""
You are an expert assistant providing comprehensive answers using company documents and general knowledge.

COMPANY DOCUMENT CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Prioritize and use ALL available information from the company documents
- Provide a COMPLETE and THOROUGH answer, not just a brief summary
- Structure your response with clear sections and details
- Include specific policies, numbers, dates, and procedures from the documents
- Supplement with relevant general knowledge only when helpful and clearly labeled as such
- Make your answer comprehensive enough to fully address the user's question
- Use bullet points or sections to organize complex information
- Distinguish clearly between company-specific information and general knowledge

RESPONSE FORMAT:
**[Main Topic] - Company Policy**

[Comprehensive answer using all available document information]

*Additional Context (if needed):*
[Any relevant general knowledge clearly marked]

RESPONSE:
"""
        
        messages = [Message(role=MessageRole.USER, content=hybrid_prompt, timestamp=None)]
        return await self.ai_provider.generate_response(messages)
    
    async def _generate_general_response(self, query: str) -> str:
        """Generate general knowledge response when documents are insufficient."""
        general_prompt = f"""
The document search didn't find sufficient relevant information for this question. 
Provide a helpful general knowledge response.

QUESTION: {query}

Note: This response is based on general knowledge rather than company-specific documents.

RESPONSE:
"""
        
        messages = [Message(role=MessageRole.USER, content=general_prompt, timestamp=None)]
        return await self.ai_provider.generate_response(messages)
    
    def _combine_retrieval_results(self, results: List[RetrievalResult]) -> str:
        """Combine multiple retrieval results into unified context."""
        # Remove duplicates and sort by confidence
        unique_results = []
        seen_texts = set()
        
        for result in sorted(results, key=lambda r: r.similarity_score, reverse=True):
            if result.chunk.text not in seen_texts:
                unique_results.append(result)
                seen_texts.add(result.chunk.text)
        
        # Combine top results
        context_parts = []
        for result in unique_results[:8]:  # Top 8 unique results
            if result.confidence_level in ["high", "medium"]:
                context_parts.append(result.chunk.text)
        
        return "\n\n---\n\n".join(context_parts)
    
    def get_agent_reasoning(self) -> List[Dict[str, Any]]:
        """Get the agent's reasoning steps for debugging."""
        return [
            {
                "step": i + 1,
                "thought_type": step.thought_type.value,
                "confidence": step.confidence_score,
                "reasoning": step.reasoning,
                "input": step.input_data,
                "output": step.output_data
            }
            for i, step in enumerate(self.agent_steps)
        ]
"""
Advanced Document Processing Pipeline for RAG System.

Implements recursive character text splitting with proper overlap,
metadata extraction, and token-aware chunking strategies.
"""

import os
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import tiktoken


@dataclass
class DocumentChunk:
    """Represents a chunk of processed document text with enhanced metadata."""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str
    source_document: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    chunk_index: int = 0
    token_count: int = 0
    confidence_score: float = 1.0


class RecursiveCharacterTextSplitter:
    """
    Implements recursive character text splitting - RAG's Gold Standard.
    
    Features:
    - Token-aware chunking (300-500 tokens)
    - 10-20% overlap for context preservation
    - Hierarchical splitting (paragraphs -> sentences -> words)
    - Section-aware metadata extraction
    """
    
    def __init__(
        self, 
        chunk_size: int = 400,  # Target 300-500 tokens
        chunk_overlap: int = 60,  # 15% of 400 tokens
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the recursive character text splitter.
        
        Args:
            chunk_size: Target chunk size in tokens (300-500 recommended)
            chunk_overlap: Overlap size in tokens (10-20% of chunk_size)
            separators: Custom separators for hierarchical splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Hierarchical separators for recursive splitting
        self.separators = separators or [
            "\n## ",     # Major sections - but allow combination
            "\n### ",    # Subsections - but allow combination  
            "\n\n",      # Double newline (paragraphs)
            "\n",        # Single newline
            ". ",        # Sentence endings
            "! ",        # Exclamations
            "? ",        # Questions
            "; ",        # Semicolons
            ", ",        # Commas
            " ",         # Spaces (words)
            ""           # Character level (last resort)
        ]
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            self.logger.warning(f"Could not load tiktoken encoder: {e}")
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback approximation: 1 token ≈ 4 characters
            return len(text) // 4
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text using recursive character splitting.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        return self._split_text_recursive(text, self.separators)
    
    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using hierarchical separators."""
        final_chunks = []
        
        # Base case: if text is small enough, return as-is
        if self.count_tokens(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        # Try each separator in order
        separator = separators[0] if separators else ""
        
        if separator == "":
            # Character-level splitting (last resort)
            return self._split_by_characters(text)
        
        # Split by current separator
        splits = text.split(separator)
        
        # If we got good splits, process them
        if len(splits) > 1:
            good_splits = []
            current_doc = ""
            
            for split in splits:
                test_doc = current_doc + (separator if current_doc else "") + split
                
                if self.count_tokens(test_doc) <= self.chunk_size:
                    current_doc = test_doc
                else:
                    if current_doc:
                        good_splits.append(current_doc)
                        current_doc = split
                    else:
                        # Single split is too large, recursively split it
                        if separators[1:]:
                            good_splits.extend(
                                self._split_text_recursive(split, separators[1:])
                            )
                        else:
                            good_splits.extend(self._split_by_characters(split))
            
            if current_doc:
                good_splits.append(current_doc)
            
            # Add overlaps between chunks
            return self._add_overlaps(good_splits)
        
        else:
            # No good splits with current separator, try next one
            if separators[1:]:
                return self._split_text_recursive(text, separators[1:])
            else:
                return self._split_by_characters(text)
    
    def _split_by_characters(self, text: str) -> List[str]:
        """Split text by characters when all else fails."""
        # Approximate character count for target tokens
        char_size = self.chunk_size * 4  # 1 token ≈ 4 chars
        
        chunks = []
        for i in range(0, len(text), char_size):
            chunk = text[i:i + char_size]
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def _add_overlaps(self, chunks: List[str]) -> List[str]:
        """Add overlaps between consecutive chunks."""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Get overlap from previous chunk
                prev_chunk = chunks[i - 1]
                overlap_text = self._get_overlap_text(prev_chunk, self.chunk_overlap)
                
                # Combine overlap with current chunk
                combined_chunk = overlap_text + " " + chunk if overlap_text else chunk
                overlapped_chunks.append(combined_chunk)
        
        return overlapped_chunks
    
    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Extract overlap text from the end of a chunk."""
        if not self.tokenizer:
            # Fallback: use last N characters
            char_overlap = overlap_tokens * 4
            return text[-char_overlap:] if len(text) > char_overlap else text
        
        # Token-based overlap
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= overlap_tokens:
            return text
        
        overlap_tokens_list = tokens[-overlap_tokens:]
        return self.tokenizer.decode(overlap_tokens_list)


class DocumentProcessor:
    """
    Enhanced document processor using recursive character text splitting.
    
    Features:
    - Section-aware processing
    - Enhanced metadata extraction
    - Multiple format support
    - Token-aware chunking
    """
    
    def __init__(
        self, 
        chunk_size: int = 400, 
        chunk_overlap: int = 60,
        min_chunk_size: int = 50
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Target chunk size in tokens (300-500)
            chunk_overlap: Overlap in tokens (10-20% of chunk_size)
            min_chunk_size: Minimum chunk size to keep
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.logger = logging.getLogger(__name__)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def process_document(self, file_path: str) -> List[DocumentChunk]:
        """
        Process a document and return enhanced chunks.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of document chunks with metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension in ['.txt', '.md']:
            return self._process_text_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _process_text_file(self, file_path: str) -> List[DocumentChunk]:
        """Process text files with section awareness."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return self._create_enhanced_chunks(content, file_path)
            
        except Exception as e:
            self.logger.error(f"Error processing text file {file_path}: {e}")
            raise
    
    def _create_enhanced_chunks(self, text: str, source_path: str) -> List[DocumentChunk]:
        """Create chunks with enhanced metadata and section awareness."""
        if not text.strip():
            return []
        
        # Clean text
        text = self._clean_text(text)
        
        # Extract sections with titles
        sections = self._extract_sections(text)
        
        all_chunks = []
        chunk_counter = 0
        
        for section_title, section_text in sections:
            # Split section into chunks
            text_chunks = self.text_splitter.split_text(section_text)
            
            for chunk_text in text_chunks:
                if self.text_splitter.count_tokens(chunk_text) < self.min_chunk_size:
                    continue
                
                chunk = DocumentChunk(
                    text=chunk_text,
                    metadata={
                        "source": Path(source_path).name,
                        "source_path": source_path,
                        "section_title": section_title,
                        "chunk_index": chunk_counter,
                        "token_count": self.text_splitter.count_tokens(chunk_text),
                        "processing_method": "recursive_character_splitting"
                    },
                    chunk_id=f"{Path(source_path).stem}_s{len(all_chunks)}_c{chunk_counter}",
                    source_document=source_path,
                    section_title=section_title,
                    chunk_index=chunk_counter,
                    token_count=self.text_splitter.count_tokens(chunk_text)
                )
                
                all_chunks.append(chunk)
                chunk_counter += 1
        
        self.logger.info(f"Created {len(all_chunks)} chunks from {source_path}")
        return all_chunks
    
    def _extract_sections(self, text: str) -> List[Tuple[str, str]]:
        """Extract sections with titles from text."""
        sections = []
        
        # Pattern to match markdown-style headers
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = text.split('\n')
        
        current_section_title = "Introduction"
        current_section_content = []
        
        for line in lines:
            header_match = re.match(header_pattern, line.strip())
            
            if header_match:
                # Save previous section
                if current_section_content:
                    sections.append((
                        current_section_title,
                        '\n'.join(current_section_content).strip()
                    ))
                
                # Start new section
                current_section_title = header_match.group(2).strip()
                current_section_content = []
            else:
                current_section_content.append(line)
        
        # Add final section
        if current_section_content:
            sections.append((
                current_section_title,
                '\n'.join(current_section_content).strip()
            ))
        
        # If no sections found, treat whole text as one section
        if not sections:
            sections = [("Document Content", text)]
        
        return sections
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace while preserving structure
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Max 2 consecutive newlines
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces/tabs
        
        # Remove problematic characters
        text = text.replace('\x00', ' ')  # Remove null bytes
        text = text.replace('\f', ' ')    # Remove form feeds
        text = text.replace('\r', '')     # Remove carriage returns
        
        return text.strip()
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return ['.txt', '.md']
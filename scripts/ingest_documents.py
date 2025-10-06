"""
Enhanced Document Ingestion Script.

Uses industry-standard recursive character text splitting
for optimal document chunking and retrieval.

Usage:
    python scripts/enhanced_ingest.py
"""

import sys
import logging
from pathlib import Path
from typing import List

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import Configuration
from src.rag.minilm_embeddings import MiniLMEmbeddingService
from src.rag.vector_store import EnhancedChromaVectorStore
from src.rag.document_processor import DocumentProcessor, DocumentChunk


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('enhanced_ingestion.log'),
            logging.StreamHandler()
        ]
    )


def get_document_files(data_dir: Path) -> List[Path]:
    """
    Get all document files from the data directory.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        List of document file paths
    """
    documents_dir = data_dir / "documents"
    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")
    
    # Supported file extensions
    supported_extensions = {'.txt', '.md', '.pdf', '.docx', '.doc'}
    
    document_files = []
    for ext in supported_extensions:
        document_files.extend(documents_dir.glob(f"**/*{ext}"))
    
    return sorted(document_files)


def process_documents(
    document_files: List[Path],
    processor: DocumentProcessor
) -> List[DocumentChunk]:
    """
    Process all documents using enhanced document processor.
    
    Args:
        document_files: List of document file paths
        processor: Enhanced document processor
        
    Returns:
        List of processed document chunks
    """
    all_chunks = []
    
    logger = logging.getLogger(__name__)
    
    for doc_path in document_files:
        try:
            logger.info(f"Processing: {doc_path.name}")
            
            # Process with enhanced processor
            chunks = processor.process_document(str(doc_path))
            
            all_chunks.extend(chunks)
            
            logger.info(
                f"✅ {doc_path.name}: {len(chunks)} chunks "
                f"(avg {sum(c.token_count for c in chunks) // len(chunks) if chunks else 0} tokens each)"
            )
            
        except Exception as e:
            logger.error(f"❌ Error processing {doc_path.name}: {e}")
            continue
    
    return all_chunks


def ingest_documents() -> None:
    """Main ingestion function with enhanced RAG processing."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 Starting enhanced document ingestion...")
        
        # Initialize configuration
        config = Configuration()
        
        # Initialize enhanced components
        embedding_service = MiniLMEmbeddingService(config)
        
        # Initialize enhanced document processor
        processor = DocumentProcessor(
            chunk_size=700,        # Target 700 tokens for better context
            chunk_overlap=100,     # Higher overlap for continuity
            min_chunk_size=50      # Minimum 50 tokens
        )
        
        # Initialize enhanced vector store
        vector_store = EnhancedChromaVectorStore(
            embedding_service=embedding_service,
            persist_directory="./chroma_db"
        )
        
        # Get document files
        data_dir = Path("data")
        document_files = get_document_files(data_dir)
        
        if not document_files:
            logger.warning("⚠️ No documents found in data/documents/")
            return
        
        logger.info(f"📁 Found {len(document_files)} documents to process")
        
        # Clear existing collection
        logger.info("🗑️ Clearing existing vector store...")
        vector_store.delete_collection()
        
        # Process all documents
        logger.info("📝 Processing documents with recursive text splitting...")
        all_chunks = process_documents(document_files, processor)
        
        if not all_chunks:
            logger.error("❌ No chunks generated from documents")
            return
        
        # Calculate statistics
        total_tokens = sum(chunk.token_count for chunk in all_chunks)
        avg_tokens = total_tokens // len(all_chunks)
        
        logger.info(
            f"📊 Generated {len(all_chunks)} chunks, "
            f"total tokens: {total_tokens:,}, "
            f"average: {avg_tokens} tokens per chunk"
        )
        
        # Add chunks to vector store
        logger.info("🔗 Adding chunks to enhanced vector store...")
        vector_store.add_chunks(all_chunks)
        
        # Verify ingestion
        stats = vector_store.get_collection_stats()
        logger.info(f"✅ Ingestion complete! {stats['total_chunks']} chunks in vector store")
        
        # Test retrieval with sample queries
        logger.info("🧪 Testing enhanced retrieval...")
        test_queries = [
            "contact information",
            "company policy",
            "benefits"
        ]
        
        for query in test_queries:
            try:
                rag_strategy = vector_store.get_rag_strategy(query, max_results=3)
                confidence = rag_strategy["confidence"]
                max_sim = rag_strategy["max_similarity"]
                use_rag = rag_strategy["use_rag"]
                
                logger.info(
                    f"Query '{query}': "
                    f"confidence={confidence}, "
                    f"similarity={max_sim:.3f}, "
                    f"use_rag={use_rag}"
                )
            except Exception as e:
                logger.warning(f"Test query '{query}' failed: {e}")
        
        logger.info("🎉 Enhanced ingestion completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    ingest_documents()
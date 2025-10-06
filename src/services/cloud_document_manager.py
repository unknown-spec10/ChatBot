"""
Cloud Document Initialization

This script initializes documents in the vector store for Streamlit Cloud deployment.
Since Streamlit Cloud has an ephemeral file system, we need to reinitialize
documents on each app start.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional
from src.rag.document_processor import DocumentProcessor
from src.rag.minilm_embeddings import MiniLMEmbeddingService
from src.rag.vector_store import EnhancedChromaVectorStore
from src.config.settings import Configuration


class CloudDocumentManager:
    """Manages document initialization for cloud deployment."""
    
    def __init__(self, config: Configuration):
        """Initialize the cloud document manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def initialize_documents(self, documents_path: str = "./data/documents") -> bool:
        """
        Initialize documents in the vector store for cloud deployment.
        
        Args:
            documents_path: Path to documents directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if documents exist
            doc_dir = Path(documents_path)
            if not doc_dir.exists():
                self.logger.warning(f"Documents directory not found: {documents_path}")
                return False
            
            # Get list of supported document files
            supported_extensions = {'.txt', '.pdf'}
            document_files = [
                f for f in doc_dir.rglob("*") 
                if f.is_file() and f.suffix.lower() in supported_extensions
            ]
            
            if not document_files:
                self.logger.warning(f"No supported documents found in {documents_path}")
                return False
            
            self.logger.info(f"Found {len(document_files)} documents to process")
            
            # Initialize RAG components
            embedding_service = MiniLMEmbeddingService(self.config)
            vector_store = EnhancedChromaVectorStore(
                embedding_service=embedding_service,
                persist_directory="./chroma_db"
            )
            
            # Check if documents are already loaded
            stats = vector_store.get_collection_stats()
            if stats.get('total_chunks', 0) > 0:
                self.logger.info(f"Documents already loaded: {stats['total_chunks']} chunks")
                return True
            
            # Initialize document processor
            processor = DocumentProcessor(
                chunk_size=self.config.get_chunk_size(),
                chunk_overlap=self.config.get_chunk_overlap()
            )
            
            # Process each document
            total_chunks = 0
            for doc_file in document_files:
                try:
                    self.logger.info(f"Processing: {doc_file.name}")
                    
                    # Read and process document
                    with open(doc_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    chunks = processor.process_document(content, str(doc_file))
                    
                    if chunks:
                        # Store chunks in vector store
                        vector_store.add_documents(chunks)
                        total_chunks += len(chunks)
                        self.logger.info(f"Added {len(chunks)} chunks from {doc_file.name}")
                    else:
                        self.logger.warning(f"No chunks generated from {doc_file.name}")
                        
                except Exception as e:
                    self.logger.error(f"Error processing {doc_file}: {str(e)}")
                    continue
            
            if total_chunks > 0:
                self.logger.info(f"Successfully initialized {total_chunks} document chunks")
                return True
            else:
                self.logger.warning("No documents were successfully processed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error initializing documents: {str(e)}")
            return False
    
    def get_document_status(self) -> dict:
        """Get current document status for display."""
        try:
            embedding_service = MiniLMEmbeddingService(self.config)
            vector_store = EnhancedChromaVectorStore(
                embedding_service=embedding_service,
                persist_directory="./chroma_db"
            )
            
            stats = vector_store.get_collection_stats()
            return {
                "loaded": stats.get('total_chunks', 0) > 0,
                "total_chunks": stats.get('total_chunks', 0),
                "status": "ready" if stats.get('total_chunks', 0) > 0 else "empty"
            }
        except Exception as e:
            return {
                "loaded": False,
                "total_chunks": 0,
                "status": f"error: {str(e)}"
            }


def initialize_cloud_documents(config: Optional[Configuration] = None) -> bool:
    """
    Convenience function to initialize documents for cloud deployment.
    
    Args:
        config: Optional configuration instance
        
    Returns:
        True if successful, False otherwise
    """
    if config is None:
        config = Configuration()
    
    manager = CloudDocumentManager(config)
    return manager.initialize_documents()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize documents
    config = Configuration()
    success = initialize_cloud_documents(config)
    
    if success:
        print("✅ Documents initialized successfully")
    else:
        print("❌ Failed to initialize documents")
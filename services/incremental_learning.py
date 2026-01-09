"""
Incremental learning service for updating vector store without full rebuild.
Supports adding new documents and tracking changes over time.
"""
import os
import logging
import pickle
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class IncrementalLearningManager:
    """Manages incremental updates to the vector store."""
    
    def __init__(self, index_path: str, embeddings: Optional[HuggingFaceEmbeddings] = None):
        """
        Initialize incremental learning manager.
        
        Args:
            index_path: Path to FAISS index directory
            embeddings: Embeddings model (will create if not provided)
        """
        self.index_path = index_path
        self.metadata_path = os.path.join(index_path, "incremental_metadata.pkl")
        
        if embeddings is None:
            model_path = "./all-MiniLM-L6-v2"
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={'device': 'cpu'}
            )
        else:
            self.embeddings = embeddings
        
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load incremental learning metadata."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
                return self._create_default_metadata()
        return self._create_default_metadata()
    
    def _save_metadata(self):
        """Save incremental learning metadata."""
        try:
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def _create_default_metadata(self) -> Dict[str, Any]:
        """Create default metadata structure."""
        return {
            "version": 1,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_documents": 0,
            "document_sources": {},  # file_path -> {added_at, num_docs, version}
            "update_history": []
        }
    
    def add_documents(
        self,
        documents: List[Document],
        source_file: Optional[str] = None,
        department: Optional[str] = None,
        document_type: Optional[str] = None
    ) -> bool:
        """
        Add new documents to existing vector store incrementally.
        
        Args:
            documents: List of documents to add
            source_file: Optional source file path for tracking
            department: Department classification for this source
            document_type: Document type classification for this source
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not documents:
                logger.warning("No documents to add")
                return False
            
            # Load existing vector store
            if not os.path.exists(self.index_path):
                logger.error(f"Index path does not exist: {self.index_path}")
                return False
            
            vectorstore = FAISS.load_local(
                self.index_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Add new documents
            vectorstore.add_documents(documents)
            
            # Save updated vector store
            vectorstore.save_local(self.index_path)
            
            # Update metadata
            self.metadata["last_updated"] = datetime.now().isoformat()
            self.metadata["total_documents"] += len(documents)
            
            if source_file:
                source_metadata = {
                    "added_at": datetime.now().isoformat(),
                    "num_docs": len(documents),
                    "version": self.metadata["version"]
                }
                if department:
                    source_metadata["department"] = department
                if document_type:
                    source_metadata["document_type"] = document_type
                
                self.metadata["document_sources"][source_file] = source_metadata
            
            self.metadata["update_history"].append({
                "timestamp": datetime.now().isoformat(),
                "action": "add_documents",
                "num_documents": len(documents),
                "source_file": source_file
            })
            
            self._save_metadata()
            
            logger.info(f"✓ Successfully added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"✗ Error adding documents: {e}")
            return False
    
    def remove_documents_by_source(self, source_file: str) -> bool:
        """
        Remove documents from a specific source file.
        Note: FAISS doesn't support direct deletion, so this rebuilds without those docs.
        
        Args:
            source_file: Source file path to remove
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Removing documents from source: {source_file}")
            
            # Load existing vector store
            vectorstore = FAISS.load_local(
                self.index_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Get all documents
            all_docs = vectorstore.docstore._dict
            
            # Filter out documents from the specified source
            filtered_docs = [
                doc for doc in all_docs.values()
                if doc.metadata.get("source") != os.path.basename(source_file)
                and doc.metadata.get("file_path") != source_file
            ]
            
            if len(filtered_docs) == len(all_docs):
                logger.warning(f"No documents found from source: {source_file}")
                return False
            
            # Rebuild vector store with filtered documents
            new_vectorstore = FAISS.from_documents(
                filtered_docs,
                embedding=self.embeddings
            )
            
            # Save updated vector store
            new_vectorstore.save_local(self.index_path)
            
            # Update metadata
            removed_count = len(all_docs) - len(filtered_docs)
            self.metadata["last_updated"] = datetime.now().isoformat()
            self.metadata["total_documents"] -= removed_count
            
            if source_file in self.metadata["document_sources"]:
                del self.metadata["document_sources"][source_file]
            
            self.metadata["update_history"].append({
                "timestamp": datetime.now().isoformat(),
                "action": "remove_documents",
                "num_documents": removed_count,
                "source_file": source_file
            })
            
            self._save_metadata()
            
            logger.info(f"✓ Successfully removed {removed_count} documents from {source_file}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Error removing documents: {e}")
            return False
    
    def update_documents_from_source(self, documents: List[Document], source_file: str) -> bool:
        """
        Update documents from a specific source (remove old, add new).
        
        Args:
            documents: New documents to add
            source_file: Source file path
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Updating documents from source: {source_file}")
        
        # Remove old documents from this source
        self.remove_documents_by_source(source_file)
        
        # Add new documents
        return self.add_documents(documents, source_file)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "total_documents": self.metadata["total_documents"],
            "num_sources": len(self.metadata["document_sources"]),
            "created_at": self.metadata["created_at"],
            "last_updated": self.metadata["last_updated"],
            "version": self.metadata["version"],
            "num_updates": len(self.metadata["update_history"])
        }
    
    def get_source_info(self, source_file: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific source file."""
        return self.metadata["document_sources"].get(source_file)
    
    def list_sources(self) -> List[str]:
        """List all source files in the vector store."""
        return list(self.metadata["document_sources"].keys())
    
    def get_update_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent update history."""
        return self.metadata["update_history"][-limit:]


def create_or_update_index(
    documents: List[Document],
    index_path: str,
    embeddings: Optional[HuggingFaceEmbeddings] = None,
    incremental: bool = True,
    source_file: Optional[str] = None,
    department: Optional[str] = None,
    document_type: Optional[str] = None
) -> bool:
    """
    Create new index or update existing one incrementally.
    
    Args:
        documents: Documents to add
        index_path: Path to FAISS index
        embeddings: Embeddings model
        incremental: If True and index exists, add incrementally; if False, rebuild
        source_file: Source file path for metadata tracking
        department: Department classification
        document_type: Document type classification
    
    Returns:
        True if successful
    """
    try:
        if embeddings is None:
            model_path = "./all-MiniLM-L6-v2"
            embeddings = HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={'device': 'cpu'}
            )
        
        # Check if index exists
        index_exists = os.path.exists(index_path) and os.path.exists(os.path.join(index_path, "index.faiss"))
        
        if index_exists and incremental:
            # Incremental update
            logger.info("Updating existing index incrementally...")
            manager = IncrementalLearningManager(index_path, embeddings)
            return manager.add_documents(
                documents, 
                source_file=source_file,
                department=department,
                document_type=document_type
            )
        else:
            # Create new index
            logger.info("Creating new index...")
            vectorstore = FAISS.from_documents(documents, embedding=embeddings)
            vectorstore.save_local(index_path)
            
            # Initialize metadata
            manager = IncrementalLearningManager(index_path, embeddings)
            manager.metadata["total_documents"] = len(documents)
            manager._save_metadata()
            
            logger.info(f"✓ Created new index with {len(documents)} documents")
            return True
            
    except Exception as e:
        logger.error(f"✗ Error creating/updating index: {e}")
        return False


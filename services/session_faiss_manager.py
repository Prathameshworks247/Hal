"""
Session FAISS Manager - Core logic for managing ephemeral session-specific FAISS indexes.
Handles conversation memory and user-uploaded file embeddings per session.
"""
import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from models.session_models import SessionMetadata, SessionMemoryEntry, SessionConfig
from services.session_storage import (
    get_session_directory,
    get_session_faiss_path,
    get_session_metadata_path,
    create_session_directory,
    save_session_metadata,
    load_session_metadata,
    session_exists,
    session_has_faiss,
    delete_session
)

logger = logging.getLogger(__name__)

# In-memory cache for loaded FAISS indexes
_session_faiss_cache: Dict[str, FAISS] = {}


class SessionFAISSManager:
    """
    Manages ephemeral session-specific FAISS indexes.
    
    Each session has:
    - Uploaded file embeddings (if user uploaded a file)
    - Conversation memory embeddings (query + response pairs)
    """
    
    def __init__(self, session_id: str, embeddings: HuggingFaceEmbeddings, config: Optional[SessionConfig] = None):
        """
        Initialize SessionFAISSManager.
        
        Args:
            session_id: Unique session identifier
            embeddings: HuggingFace embeddings model
            config: Optional session configuration
        """
        self.session_id = session_id
        self.embeddings = embeddings
        self.config = config or SessionConfig()
        
        # Paths
        self.session_dir = get_session_directory(session_id)
        self.faiss_path = get_session_faiss_path(session_id)
        self.metadata_path = get_session_metadata_path(session_id)
        
        # Initialize session if it doesn't exist
        if not session_exists(session_id):
            self._initialize_session()
        
        # Load metadata
        self.metadata = load_session_metadata(session_id)
        if not self.metadata:
            self.metadata = SessionMetadata(session_id=session_id)
            save_session_metadata(session_id, self.metadata)
        
        # Lazy-loaded FAISS index
        self._faiss_index: Optional[FAISS] = None
    
    def _initialize_session(self):
        """Create new session directory structure."""
        try:
            create_session_directory(self.session_id)
            
            # Create initial metadata
            metadata = SessionMetadata(session_id=self.session_id)
            save_session_metadata(self.session_id, metadata)
            
            logger.info(f"Initialized new session: {self.session_id}")
            
        except Exception as e:
            logger.error(f"Error initializing session {self.session_id}: {str(e)}")
            raise
    
    def add_uploaded_file_embeddings(self, documents: List[Document], source_file: Optional[str] = None) -> bool:
        """
        Embed user's uploaded file(s) and save to session FAISS.
        Supports incremental addition: appends if FAISS exists, creates new if not.
        Can be called multiple times for different files in the same session.
        
        Args:
            documents: List of document chunks from parsed file(s)
            source_file: Original filename for tracking (optional, extracted from metadata if not provided)
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Adding embeddings for {len(documents)} document chunks (session: {self.session_id}, source: {source_file})")
            
            if not documents:
                logger.error("No documents provided for embedding")
                return False
            
            # Validate documents have content
            valid_documents = []
            for idx, doc in enumerate(documents):
                if not doc.page_content or not doc.page_content.strip():
                    logger.warning(f"Document {idx} has empty content, skipping")
                    continue
                valid_documents.append(doc)
            
            if not valid_documents:
                logger.error("No valid documents with content found")
                return False
            
            logger.info(f"Valid documents: {len(valid_documents)}/{len(documents)}")
            
            # Extract source_file from metadata if not provided
            if not source_file and valid_documents:
                source_file = valid_documents[0].metadata.get("source") or valid_documents[0].metadata.get("source_file")
            
            # Ensure all documents have required metadata
            for doc in valid_documents:
                # Mark as uploaded file content
                doc.metadata["type"] = "uploaded_file"
                doc.metadata["authoritative"] = True
                doc.metadata["session_id"] = self.session_id
                
                # Set source_file if not already set
                if source_file and "source_file" not in doc.metadata:
                    doc.metadata["source_file"] = source_file
                
                # Ensure source_file is in metadata (use source as fallback)
                if "source_file" not in doc.metadata and "source" in doc.metadata:
                    doc.metadata["source_file"] = doc.metadata["source"]
            
            # Check if FAISS index already exists for this session
            existing_faiss = self.load_session_faiss()
            
            if existing_faiss is None:
                # Create new FAISS index
                logger.info(f"Creating new FAISS index with {len(valid_documents)} documents...")
                try:
                    faiss_index = FAISS.from_documents(valid_documents, self.embeddings)
                    logger.info(f"✓ New FAISS index created with {len(valid_documents)} documents")
                except Exception as e:
                    logger.exception(f"Error creating new FAISS index: {str(e)}")
                    raise
            else:
                # Append to existing FAISS index
                logger.info(f"Appending {len(valid_documents)} documents to existing FAISS index...")
                try:
                    existing_faiss.add_documents(valid_documents)
                    faiss_index = existing_faiss
                    logger.info(f"✓ Appended {len(valid_documents)} documents to existing FAISS index")
                except Exception as e:
                    logger.exception(f"Error appending to existing FAISS index: {str(e)}")
                    raise
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.faiss_path), exist_ok=True)
            
            # Save updated index to disk
            try:
                faiss_index.save_local(self.faiss_path)
                logger.info(f"✓ Saved session FAISS to: {self.faiss_path}")
            except Exception as e:
                logger.exception(f"Error saving FAISS index to disk: {str(e)}")
                raise
            
            # Update cache
            if self.config.cache_in_memory:
                _session_faiss_cache[self.session_id] = faiss_index
                logger.debug(f"Updated in-memory cache for session {self.session_id}")
            
            self._faiss_index = faiss_index
            
            # Update metadata
            self.metadata.has_uploaded_file = True
            self.metadata.document_chunks += len(valid_documents)
            self.metadata.total_embeddings += len(valid_documents)
            
            # Track uploaded file information
            file_info = {
                "file_name": source_file,
                "file_type": valid_documents[0].metadata.get("file_type"),
                "chunks": len(valid_documents),
                "uploaded_at": datetime.now().isoformat()
            }
            
            # Check if file already tracked (avoid duplicates)
            file_exists = False
            for existing_file in self.metadata.uploaded_files:
                if existing_file.get("file_name") == source_file:
                    # Update existing entry
                    existing_file.update(file_info)
                    file_exists = True
                    break
            
            if not file_exists:
                self.metadata.uploaded_files.append(file_info)
            
            # Legacy fields: set to first file for backward compatibility
            if not self.metadata.uploaded_file_name and source_file:
                self.metadata.uploaded_file_name = source_file
            if not self.metadata.uploaded_file_type and file_info.get("file_type"):
                self.metadata.uploaded_file_type = file_info.get("file_type")
            
            save_session_metadata(self.session_id, self.metadata)
            
            logger.info(f"✓ Uploaded file embedded: {source_file} ({len(valid_documents)} chunks) in session {self.session_id}")
            logger.info(f"  Total files in session: {len(self.metadata.uploaded_files)}, Total chunks: {self.metadata.document_chunks}")
            return True
            
        except Exception as e:
            logger.exception(f"Error adding uploaded file embeddings: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def load_session_faiss(self) -> Optional[FAISS]:
        """
        Load session FAISS from disk (or cache).
        Loads pre-computed embeddings - NO re-embedding.
        
        Returns:
            FAISS index or None if doesn't exist
        """
        try:
            # Check cache first
            if self.config.cache_in_memory and self.session_id in _session_faiss_cache:
                logger.debug(f"Loading session FAISS from cache: {self.session_id}")
                return _session_faiss_cache[self.session_id]
            
            # Check if FAISS exists on disk
            if not session_has_faiss(self.session_id):
                logger.debug(f"No FAISS index found for session: {self.session_id}")
                return None
            
            # Load from disk
            logger.debug(f"Loading session FAISS from disk: {self.faiss_path}")
            faiss_index = FAISS.load_local(
                self.faiss_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Update cache
            if self.config.cache_in_memory:
                _session_faiss_cache[self.session_id] = faiss_index
            
            self._faiss_index = faiss_index
            return faiss_index
            
        except Exception as e:
            logger.error(f"Error loading session FAISS: {str(e)}")
            return None
    
    def add_conversation_memory(self, query: str, response: str) -> bool:
        """
        Add conversation turn (Q&A pair) to session FAISS.
        Creates embedding and stores in SESSION_FAISS.
        
        Args:
            query: User's query
            response: Assistant's response
            
        Returns:
            True if successful
        """
        try:
            # Check memory size limit
            if self.metadata.conversation_turns >= self.config.max_memory_size:
                logger.warning(f"Session {self.session_id} reached max memory size, skipping")
                return False
            
            # Create conversation document
            conversation_text = f"User Query: {query}\n\nAssistant Response: {response}"
            
            conversation_doc = Document(
                page_content=conversation_text,
                metadata={
                    "type": "conversation_memory",
                    "authoritative": False,  # Non-authoritative
                    "confidence": "low",
                    "turn_number": self.metadata.conversation_turns + 1,
                    "timestamp": datetime.now().isoformat(),
                    "session_id": self.session_id
                }
            )
            
            # Load existing FAISS or create new
            faiss_index = self.load_session_faiss()
            
            if faiss_index is None:
                # First conversation - create new index
                logger.info(f"Creating new session FAISS with first conversation (session: {self.session_id})")
                faiss_index = FAISS.from_documents([conversation_doc], self.embeddings)
            else:
                # Add to existing index
                logger.debug(f"Adding conversation memory to existing FAISS (session: {self.session_id})")
                faiss_index.add_documents([conversation_doc])
            
            # Save updated index to disk
            faiss_index.save_local(self.faiss_path)
            
            # Update cache
            if self.config.cache_in_memory:
                _session_faiss_cache[self.session_id] = faiss_index
            
            self._faiss_index = faiss_index
            
            # Update metadata
            self.metadata.conversation_turns += 1
            self.metadata.total_embeddings += 1
            save_session_metadata(self.session_id, self.metadata)
            
            logger.debug(f"✓ Conversation memory added (turn {self.metadata.conversation_turns})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding conversation memory: {str(e)}")
            return False
    
    def retrieve_from_session(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve documents from SESSION_FAISS only.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of retrieved documents
        """
        try:
            faiss_index = self.load_session_faiss()
            
            if faiss_index is None:
                logger.debug(f"No FAISS index for session {self.session_id}")
                return []
            
            # Search session FAISS
            results = faiss_index.similarity_search(query, k=k)
            
            logger.debug(f"Retrieved {len(results)} documents from session FAISS")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving from session: {str(e)}")
            return []
    
    def retrieve_from_session_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """
        Retrieve documents with similarity scores from SESSION_FAISS.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        try:
            faiss_index = self.load_session_faiss()
            
            if faiss_index is None:
                return []
            
            # Search with scores
            results = faiss_index.similarity_search_with_score(query, k=k)
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving with scores from session: {str(e)}")
            return []
    
    def get_session_metadata(self) -> SessionMetadata:
        """Get current session metadata."""
        return self.metadata
    
    def destroy_session(self) -> bool:
        """
        Delete session FAISS, files, and metadata.
        
        Returns:
            True if successful
        """
        try:
            # Remove from cache
            if self.session_id in _session_faiss_cache:
                del _session_faiss_cache[self.session_id]
            
            # Delete from disk
            success = delete_session(self.session_id)
            
            if success:
                logger.info(f"✓ Destroyed session: {self.session_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error destroying session: {str(e)}")
            return False
    
    @staticmethod
    def merge_retrieval_results(
        global_results: List[Document],
        session_results: List[Document],
        k: int = 5,
        prioritize: str = "documents"
    ) -> List[Document]:
        """
        Merge results from GLOBAL_FAISS and SESSION_FAISS.
        
        Authority rules:
        - Document context (authoritative=True) > conversation memory (authoritative=False)
        - Prioritize based on 'prioritize' parameter
        
        Args:
            global_results: Results from GLOBAL_FAISS
            session_results: Results from SESSION_FAISS
            k: Total number of results to return
            prioritize: "documents" or "balanced"
            
        Returns:
            Merged and sorted list of documents
        """
        # Separate by authority
        authoritative = []
        non_authoritative = []
        
        for doc in global_results + session_results:
            is_auth = doc.metadata.get("authoritative", True)
            
            if is_auth:
                authoritative.append(doc)
            else:
                non_authoritative.append(doc)
        
        # Prioritize authoritative documents
        if prioritize == "documents":
            # Mostly authoritative, some conversation memory
            auth_count = min(k - 1, len(authoritative))
            non_auth_count = k - auth_count
            
            merged = authoritative[:auth_count] + non_authoritative[:non_auth_count]
        else:
            # Balanced mix
            merged = authoritative + non_authoritative
        
        return merged[:k]
    
    def has_uploaded_file(self) -> bool:
        """Check if session has an uploaded file."""
        return self.metadata.has_uploaded_file
    
    def get_conversation_count(self) -> int:
        """Get number of conversation turns in this session."""
        return self.metadata.conversation_turns
    
    def clear_cache(self):
        """Clear session from memory cache."""
        if self.session_id in _session_faiss_cache:
            del _session_faiss_cache[self.session_id]
            logger.debug(f"Cleared cache for session: {self.session_id}")


def clear_all_session_cache():
    """Clear all sessions from memory cache."""
    global _session_faiss_cache
    _session_faiss_cache.clear()
    logger.info("Cleared all session FAISS cache")


def get_cached_session_ids() -> List[str]:
    """Get list of session IDs currently in memory cache."""
    return list(_session_faiss_cache.keys())


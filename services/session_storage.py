"""
Session storage management for ephemeral FAISS indexes.
Handles filesystem operations for session directories.
"""
import os
import json
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from models.session_models import SessionMetadata, SessionInfo

logger = logging.getLogger(__name__)

# Base directory for all sessions (relative to project root)
SESSION_BASE_DIR = "sessions"


def get_session_base_path() -> str:
    """
    Get absolute path to sessions directory.
    Creates directory if it doesn't exist.
    
    Returns:
        Absolute path to sessions/ directory
    """
    base_path = os.path.join(os.getcwd(), SESSION_BASE_DIR)
    os.makedirs(base_path, exist_ok=True)
    return base_path


def get_session_directory(session_id: str) -> str:
    """
    Get path to a specific session directory.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Path to session directory (e.g., sessions/session-abc123/)
    """
    session_dir = os.path.join(get_session_base_path(), f"session-{session_id}")
    return session_dir


def get_session_faiss_path(session_id: str) -> str:
    """
    Get path to session FAISS index directory.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Path to FAISS index (e.g., sessions/session-abc123/faiss_index/)
    """
    return os.path.join(get_session_directory(session_id), "faiss_index")


def get_session_metadata_path(session_id: str) -> str:
    """
    Get path to session metadata JSON file.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Path to metadata file (e.g., sessions/session-abc123/metadata.json)
    """
    return os.path.join(get_session_directory(session_id), "metadata.json")


def get_session_uploads_path(session_id: str) -> str:
    """
    Get path to session uploads directory.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Path to uploads directory (e.g., sessions/session-abc123/uploaded_files/)
    """
    return os.path.join(get_session_directory(session_id), "uploaded_files")


def create_session_directory(session_id: str) -> bool:
    """
    Create a new session directory with subdirectories.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        True if created successfully, False otherwise
    """
    try:
        session_dir = get_session_directory(session_id)
        faiss_dir = get_session_faiss_path(session_id)
        uploads_dir = get_session_uploads_path(session_id)
        
        os.makedirs(session_dir, exist_ok=True)
        os.makedirs(faiss_dir, exist_ok=True)
        os.makedirs(uploads_dir, exist_ok=True)
        
        logger.info(f"Created session directory: {session_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating session directory for {session_id}: {str(e)}")
        return False


def session_exists(session_id: str) -> bool:
    """
    Check if a session exists.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        True if session directory exists
    """
    session_dir = get_session_directory(session_id)
    return os.path.exists(session_dir)


def session_has_faiss(session_id: str) -> bool:
    """
    Check if a session has a FAISS index.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        True if FAISS index exists
    """
    faiss_path = get_session_faiss_path(session_id)
    index_file = os.path.join(faiss_path, "index.faiss")
    return os.path.exists(index_file)


def delete_session(session_id: str) -> bool:
    """
    Delete entire session directory and all contents.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        True if deleted successfully, False if not found
    """
    try:
        session_dir = get_session_directory(session_id)
        
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
            logger.info(f"Deleted session: {session_id}")
            return True
        else:
            logger.warning(f"Session not found: {session_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {str(e)}")
        return False


def list_active_sessions() -> List[str]:
    """
    List all active session IDs.
    
    Returns:
        List of session IDs (without 'session-' prefix)
    """
    try:
        sessions_dir = get_session_base_path()
        
        if not os.path.exists(sessions_dir):
            return []
        
        sessions = []
        for folder in os.listdir(sessions_dir):
            if folder.startswith("session-") and os.path.isdir(os.path.join(sessions_dir, folder)):
                # Remove 'session-' prefix
                session_id = folder.replace("session-", "")
                sessions.append(session_id)
        
        return sessions
        
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        return []


def get_session_age(session_id: str) -> Optional[float]:
    """
    Get age of session in hours.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Age in hours, or None if session doesn't exist
    """
    try:
        session_dir = get_session_directory(session_id)
        
        if not os.path.exists(session_dir):
            return None
        
        last_modified = datetime.fromtimestamp(os.path.getmtime(session_dir))
        age = datetime.now() - last_modified
        return age.total_seconds() / 3600  # Convert to hours
        
    except Exception as e:
        logger.error(f"Error getting session age for {session_id}: {str(e)}")
        return None


def cleanup_expired_sessions(max_age_hours: int = 24) -> int:
    """
    Delete sessions older than max_age_hours.
    
    Args:
        max_age_hours: Maximum age in hours before deletion
        
    Returns:
        Number of sessions deleted
    """
    try:
        sessions_dir = get_session_base_path()
        
        if not os.path.exists(sessions_dir):
            return 0
        
        deleted_count = 0
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        for session_folder in os.listdir(sessions_dir):
            session_path = os.path.join(sessions_dir, session_folder)
            
            if not os.path.isdir(session_path):
                continue
            
            # Check last modified time
            last_modified = datetime.fromtimestamp(os.path.getmtime(session_path))
            
            if last_modified < cutoff_time:
                try:
                    shutil.rmtree(session_path)
                    deleted_count += 1
                    logger.info(f"Cleaned up expired session: {session_folder}")
                except Exception as e:
                    logger.error(f"Error deleting session {session_folder}: {str(e)}")
        
        if deleted_count > 0:
            logger.info(f"Cleanup complete: Deleted {deleted_count} expired sessions")
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"Error during session cleanup: {str(e)}")
        return 0


def save_session_metadata(session_id: str, metadata: SessionMetadata) -> bool:
    """
    Save session metadata to JSON file.
    
    Args:
        session_id: Unique session identifier
        metadata: SessionMetadata object
        
    Returns:
        True if saved successfully
    """
    try:
        metadata_path = get_session_metadata_path(session_id)
        
        # Update last_accessed
        metadata.last_accessed = datetime.now()
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata.dict(), f, indent=2, default=str)
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving metadata for session {session_id}: {str(e)}")
        return False


def load_session_metadata(session_id: str) -> Optional[SessionMetadata]:
    """
    Load session metadata from JSON file.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        SessionMetadata object, or None if not found
    """
    try:
        metadata_path = get_session_metadata_path(session_id)
        
        if not os.path.exists(metadata_path):
            return None
        
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        # Convert ISO strings back to datetime
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('last_accessed'), str):
            data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        
        return SessionMetadata(**data)
        
    except Exception as e:
        logger.error(f"Error loading metadata for session {session_id}: {str(e)}")
        return None


def get_all_session_info() -> List[SessionInfo]:
    """
    Get information about all active sessions.
    
    Returns:
        List of SessionInfo objects
    """
    sessions_info = []
    
    for session_id in list_active_sessions():
        try:
            metadata = load_session_metadata(session_id)
            
            if metadata:
                age = get_session_age(session_id) or 0
                
                session_info = SessionInfo(
                    session_id=session_id,
                    created_at=metadata.created_at,
                    last_accessed=metadata.last_accessed,
                    has_uploaded_file=metadata.has_uploaded_file,
                    uploaded_file_name=metadata.uploaded_file_name,
                    conversation_turns=metadata.conversation_turns,
                    total_embeddings=metadata.total_embeddings,
                    age_hours=round(age, 2)
                )
                
                sessions_info.append(session_info)
                
        except Exception as e:
            logger.error(f"Error getting info for session {session_id}: {str(e)}")
            continue
    
    return sessions_info


def get_session_size_mb(session_id: str) -> Optional[float]:
    """
    Get total size of session directory in MB.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Size in MB, or None if session doesn't exist
    """
    try:
        session_dir = get_session_directory(session_id)
        
        if not os.path.exists(session_dir):
            return None
        
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(session_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        
        return round(total_size / (1024 * 1024), 2)  # Convert to MB
        
    except Exception as e:
        logger.error(f"Error getting size for session {session_id}: {str(e)}")
        return None


"""
Session-related Pydantic models for ephemeral FAISS memory management.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class SessionMetadata(BaseModel):
    """Metadata for a session including file and conversation info."""
    session_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)
    has_uploaded_file: bool = False
    uploaded_file_name: Optional[str] = None
    uploaded_file_type: Optional[str] = None
    conversation_turns: int = 0
    document_chunks: int = 0
    total_embeddings: int = 0
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SessionMemoryEntry(BaseModel):
    """Single conversation memory entry."""
    query: str
    response: str
    timestamp: datetime = Field(default_factory=datetime.now)
    turn_number: int
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SessionConfig(BaseModel):
    """Configuration for session management."""
    max_memory_size: int = Field(default=100, description="Maximum conversation turns to store")
    memory_ttl: int = Field(default=86400, description="Time to live in seconds (default: 24 hours)")
    auto_cleanup: bool = Field(default=True, description="Enable automatic cleanup of expired sessions")
    cache_in_memory: bool = Field(default=True, description="Cache loaded FAISS indexes in memory")


class SessionInfo(BaseModel):
    """Information about a session for API responses."""
    session_id: str
    created_at: datetime
    last_accessed: datetime
    has_uploaded_file: bool
    uploaded_file_name: Optional[str]
    conversation_turns: int
    total_embeddings: int
    age_hours: float
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


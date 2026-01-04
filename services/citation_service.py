"""
Enhanced citation and traceability service for RAG responses.
Provides detailed source attribution and confidence scoring.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def format_citation(metadata: Dict[str, Any]) -> str:
    """
    Format metadata into a human-readable citation.
    
    Args:
        metadata: Document metadata dictionary
    
    Returns:
        Formatted citation string
    """
    source = metadata.get("source", "Unknown Source")
    file_type = metadata.get("file_type", "unknown")
    
    if file_type == "pdf":
        page = metadata.get("page_number", "?")
        return f"[{source}, Page {page}]"
    
    elif file_type == "docx":
        para = metadata.get("paragraph_start", "?")
        return f"[{source}, Paragraph ~{para}]"
    
    elif file_type == "txt":
        line = metadata.get("line_start", "?")
        return f"[{source}, Line ~{line}]"
    
    elif file_type == "excel":
        row = metadata.get("row_index", "?")
        return f"[{source}, Row {row}]"
    
    else:
        return f"[{source}]"


def extract_citations_from_sources(source_documents: List[Any]) -> List[Dict[str, Any]]:
    """
    Extract citation information from source documents.
    
    Args:
        source_documents: List of retrieved documents
    
    Returns:
        List of citation dictionaries with formatted information
    """
    citations = []
    
    for idx, doc in enumerate(source_documents, start=1):
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        
        citation = {
            "citation_id": idx,
            "formatted_citation": format_citation(metadata),
            "source_file": metadata.get("source", "Unknown"),
            "file_type": metadata.get("file_type", "unknown"),
            "location": _extract_location(metadata),
            "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "ingestion_timestamp": metadata.get("ingestion_timestamp", "Unknown"),
            "full_metadata": metadata
        }
        
        citations.append(citation)
    
    return citations


def _extract_location(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Extract location information based on file type."""
    file_type = metadata.get("file_type", "unknown")
    
    if file_type == "pdf":
        return {
            "page_number": metadata.get("page_number"),
            "total_pages": metadata.get("total_pages"),
            "chunk_index": metadata.get("chunk_index")
        }
    
    elif file_type == "docx":
        return {
            "paragraph_start": metadata.get("paragraph_start"),
            "total_paragraphs": metadata.get("total_paragraphs"),
            "chunk_index": metadata.get("chunk_index")
        }
    
    elif file_type == "txt":
        return {
            "line_start": metadata.get("line_start"),
            "total_lines": metadata.get("total_lines"),
            "chunk_index": metadata.get("chunk_index")
        }
    
    elif file_type == "excel":
        return {
            "row_index": metadata.get("row_index"),
            "total_rows": metadata.get("total_rows")
        }
    
    return {}


def create_traceable_response(
    query: str,
    answer: str,
    source_documents: List[Any],
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a fully traceable response with citations and provenance.
    
    Args:
        query: Original user query
        answer: Generated answer
        source_documents: Retrieved source documents
        metadata: Additional metadata
    
    Returns:
        Comprehensive response dictionary with full traceability
    """
    citations = extract_citations_from_sources(source_documents)
    
    response = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": answer,
        "citations": citations,
        "traceability": {
            "num_sources_used": len(source_documents),
            "source_files": list(set([c["source_file"] for c in citations])),
            "file_types": list(set([c["file_type"] for c in citations])),
            "query_timestamp": datetime.now().isoformat()
        },
        "metadata": metadata or {}
    }
    
    return response


def calculate_confidence_score(source_documents: List[Any], similarity_scores: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Calculate confidence metrics for the response.
    
    Args:
        source_documents: Retrieved source documents
        similarity_scores: Optional similarity scores from retrieval
    
    Returns:
        Dictionary with confidence metrics
    """
    num_sources = len(source_documents)
    
    if similarity_scores:
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
        max_similarity = max(similarity_scores) if similarity_scores else 0
        min_similarity = min(similarity_scores) if similarity_scores else 0
    else:
        avg_similarity = 0
        max_similarity = 0
        min_similarity = 0
    
    # Determine confidence level
    if num_sources >= 3 and avg_similarity > 0.75:
        confidence_level = "high"
    elif num_sources >= 2 and avg_similarity > 0.5:
        confidence_level = "medium"
    else:
        confidence_level = "low"
    
    return {
        "confidence_level": confidence_level,
        "num_sources": num_sources,
        "avg_similarity": round(avg_similarity, 3),
        "max_similarity": round(max_similarity, 3),
        "min_similarity": round(min_similarity, 3),
        "recommendation": _get_confidence_recommendation(confidence_level, num_sources)
    }


def _get_confidence_recommendation(confidence_level: str, num_sources: int) -> str:
    """Get recommendation based on confidence level."""
    if confidence_level == "high":
        return "High confidence - Multiple relevant sources found"
    elif confidence_level == "medium":
        return "Medium confidence - Some relevant sources found, consider additional verification"
    else:
        return "Low confidence - Limited sources found, manual verification recommended"


def format_inline_citations(text: str, citations: List[Dict[str, Any]]) -> str:
    """
    Add inline citations to text (for future enhancement).
    
    Args:
        text: Response text
        citations: List of citation dictionaries
    
    Returns:
        Text with inline citations
    """
    # This is a placeholder for future inline citation functionality
    # For now, citations are provided separately
    return text


def create_audit_log_entry(
    query: str,
    response: Dict[str, Any],
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create an audit log entry for traceability.
    
    Args:
        query: User query
        response: Generated response
        user_id: Optional user identifier
        session_id: Optional session identifier
    
    Returns:
        Audit log entry dictionary
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id or "anonymous",
        "session_id": session_id or "unknown",
        "query": query,
        "num_sources": len(response.get("citations", [])),
        "source_files": response.get("traceability", {}).get("source_files", []),
        "confidence": response.get("confidence", {}).get("confidence_level", "unknown")
    }


"""
PDF Citation Service with Bounding Box Extraction.
Extracts exact text coordinates from PDFs for frontend highlighting.
"""
import logging
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# PDF libraries
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available for bbox extraction")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available")


def find_text_bbox_in_pdf(pdf_path: str, page_num: int, search_text: str, 
                          method: str = "pymupdf") -> Optional[List[float]]:
    """
    Find bounding box coordinates for specific text in a PDF page.
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page number (1-indexed)
        search_text: Text to search for
        method: "pymupdf" or "pdfplumber"
    
    Returns:
        Bbox as [x0, y0, x1, y1] in PDF coordinates (bottom-left origin), or None
    """
    if method == "pymupdf" and PYMUPDF_AVAILABLE:
        return _find_bbox_pymupdf(pdf_path, page_num, search_text)
    elif method == "pdfplumber" and PDFPLUMBER_AVAILABLE:
        return _find_bbox_pdfplumber(pdf_path, page_num, search_text)
    else:
        logger.warning(f"Method {method} not available, returning None")
        return None


def _find_bbox_pymupdf(pdf_path: str, page_num: int, search_text: str) -> Optional[List[float]]:
    """
    Find bbox using PyMuPDF (faster, more accurate).
    Returns [x0, y0, x1, y1] in PDF coordinates (bottom-left origin).
    """
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]  # Convert to 0-indexed
        
        # Search for text instances
        text_instances = page.search_for(search_text)
        
        if text_instances:
            # Use the first instance
            rect = text_instances[0]
            
            # PyMuPDF uses top-left origin, convert to bottom-left
            # rect format: (x0, y0, x1, y1) where y0 is top
            page_height = page.rect.height
            
            bbox = [
                float(rect.x0),           # Left
                float(page_height - rect.y1),  # Bottom (converted)
                float(rect.x1),           # Right
                float(page_height - rect.y0)   # Top (converted)
            ]
            
            doc.close()
            return bbox
        
        doc.close()
        return None
        
    except Exception as e:
        logger.error(f"Error finding bbox with PyMuPDF: {str(e)}")
        return None


def _find_bbox_pdfplumber(pdf_path: str, page_num: int, search_text: str) -> Optional[List[float]]:
    """
    Find bbox using pdfplumber.
    Returns [x0, y0, x1, y1] in PDF coordinates (bottom-left origin).
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num - 1]
            page_height = page.height
            
            # Extract words with coordinates
            words = page.extract_words()
            
            # Find matching text
            for word in words:
                if search_text.lower() in word['text'].lower():
                    # pdfplumber uses top-origin, convert to bottom-origin
                    bbox = [
                        float(word['x0']),                    # Left
                        float(page_height - word['bottom']),  # Bottom (converted)
                        float(word['x1']),                    # Right
                        float(page_height - word['top'])      # Top (converted)
                    ]
                    return bbox
            
            return None
            
    except Exception as e:
        logger.error(f"Error finding bbox with pdfplumber: {str(e)}")
        return None


def extract_context_around_text(pdf_path: str, page_num: int, search_text: str,
                                context_chars: int = 100) -> Optional[str]:
    """
    Extract surrounding context for a piece of text.
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page number (1-indexed)
        search_text: Text to find
        context_chars: Number of characters before/after to include
    
    Returns:
        Context string with ...text... format
    """
    try:
        if PYMUPDF_AVAILABLE:
            doc = fitz.open(pdf_path)
            page = doc[page_num - 1]
            full_text = page.get_text()
            doc.close()
            
            # Find text position
            start_idx = full_text.lower().find(search_text.lower())
            if start_idx == -1:
                return None
            
            # Extract context
            context_start = max(0, start_idx - context_chars)
            context_end = min(len(full_text), start_idx + len(search_text) + context_chars)
            
            context = full_text[context_start:context_end].strip()
            
            # Add ellipsis if truncated
            if context_start > 0:
                context = "..." + context
            if context_end < len(full_text):
                context = context + "..."
            
            return context
            
    except Exception as e:
        logger.error(f"Error extracting context: {str(e)}")
        return None


def create_pdf_citation(
    citation_id: str,
    text: str,
    pdf_path: str,
    page_num: int,
    context: Optional[str] = None,
    ref_number: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a complete PDF citation with bbox coordinates.
    
    Args:
        citation_id: Unique citation ID
        text: Quoted text from PDF
        pdf_path: Path to PDF file
        page_num: Page number (1-indexed)
        context: Optional surrounding context
        ref_number: Optional reference number like "[1]"
    
    Returns:
        Citation dictionary with bbox coordinates
    """
    # Get filename
    filename = os.path.basename(pdf_path)
    
    # Find bbox coordinates
    bbox = find_text_bbox_in_pdf(pdf_path, page_num, text)
    
    # If bbox not found, try first few words
    if not bbox and len(text.split()) > 3:
        first_words = " ".join(text.split()[:3])
        bbox = find_text_bbox_in_pdf(pdf_path, page_num, first_words)
    
    # Extract context if not provided
    if not context:
        context = extract_context_around_text(pdf_path, page_num, text)
    
    citation = {
        "id": citation_id,
        "text": text,
        "source": {
            "file": filename,
            "file_path": pdf_path,
            "page": page_num,
            "bbox": bbox if bbox else [0, 0, 0, 0]  # Fallback if not found
        }
    }
    
    if context:
        citation["context"] = context
    
    if ref_number:
        citation["ref_number"] = ref_number
    
    return citation


def extract_citations_from_retrieved_docs(
    source_documents: List[Any],
    query: str
) -> List[Dict[str, Any]]:
    """
    Extract citations with PDF bbox from retrieved documents.
    
    Args:
        source_documents: Retrieved documents from RAG
        query: Original query (for context)
    
    Returns:
        List of citation dictionaries with bbox coordinates
    """
    logger.info(f"Extracting citations from {len(source_documents)} documents")
    citations = []
    citation_counter = 1
    
    for doc in source_documents:
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        
        # Only process PDF documents
        if metadata.get("file_type") != "pdf":
            continue
        
        file_path = metadata.get("file_path")
        page_num = metadata.get("page_number")
        
        if not file_path or not page_num:
            continue
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"PDF file not found: {file_path}")
            continue
        
        # Extract text snippet (first 100 chars as citation text)
        text_snippet = doc.page_content[:100].strip()
        if len(doc.page_content) > 100:
            text_snippet += "..."
        
        # Create citation
        citation_id = f"cite_{citation_counter}"
        ref_number = f"[{citation_counter}]"
        
        citation = create_pdf_citation(
            citation_id=citation_id,
            text=text_snippet,
            pdf_path=file_path,
            page_num=page_num,
            ref_number=ref_number
        )
        
        citations.append(citation)
        citation_counter += 1
    
    return citations


def get_citation_by_id(citation_id: str, all_citations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Retrieve citation details by ID.
    
    Args:
        citation_id: Citation ID to look up
        all_citations: List of all citations
    
    Returns:
        Citation dictionary or None
    """
    for citation in all_citations:
        if citation.get("id") == citation_id:
            return citation
    return None


def create_citation_store() -> Dict[str, Any]:
    """
    Create an in-memory citation store for the session.
    In production, this should be a proper cache (Redis, etc.)
    
    Returns:
        Citation store dictionary
    """
    return {
        "citations": {},
        "created_at": datetime.now().isoformat()
    }


# Global citation store (in production, use Redis or similar)
CITATION_STORE = create_citation_store()


def store_citations(session_id: str, citations: List[Dict[str, Any]]):
    """
    Store citations for a session.
    
    Args:
        session_id: Session/query identifier
        citations: List of citation dictionaries
    """
    CITATION_STORE["citations"][session_id] = {
        "citations": citations,
        "timestamp": datetime.now().isoformat()
    }
    logger.info(f"Stored {len(citations)} citations for session {session_id}")


def get_citations_for_session(session_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve citations for a session.
    
    Args:
        session_id: Session/query identifier
    
    Returns:
        List of citations or empty list
    """
    session_data = CITATION_STORE["citations"].get(session_id)
    if session_data:
        return session_data.get("citations", [])
    return []


def test_bbox_extraction(pdf_path: str, page_num: int, text: str):
    """
    Test function to verify bbox extraction works.
    
    Args:
        pdf_path: Path to test PDF
        page_num: Page to test
        text: Text to find
    
    Returns:
        Test results dictionary
    """
    results = {
        "pdf_exists": os.path.exists(pdf_path),
        "pymupdf_available": PYMUPDF_AVAILABLE,
        "pdfplumber_available": PDFPLUMBER_AVAILABLE,
        "bbox": None,
        "context": None
    }
    
    if results["pdf_exists"]:
        results["bbox"] = find_text_bbox_in_pdf(pdf_path, page_num, text)
        results["context"] = extract_context_around_text(pdf_path, page_num, text)
    
    return results


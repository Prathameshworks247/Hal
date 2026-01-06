"""
Multi-format document parser for PDF, DOCX, TXT, and Excel files.
Supports offline operation with enhanced metadata and citation tracking.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from langchain.schema import Document

logger = logging.getLogger(__name__)

# PDF parsing
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
    PDF_LIBRARY = "pymupdf"
except ImportError:
    try:
        import pdfplumber
        PDF_AVAILABLE = True
        PDF_LIBRARY = "pdfplumber"
    except ImportError:
        PDF_AVAILABLE = False
        PDF_LIBRARY = None
        logger.warning("PDF parsing not available. Install PyMuPDF (pip install pymupdf) or pdfplumber.")

# DOCX parsing
try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logger.warning("DOCX parsing not available. Install python-docx (pip install python-docx).")

# Excel parsing (already available via pandas)
import pandas as pd


def parse_pdf(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Parse PDF file with page-level tracking and smart chunking.
    
    Args:
        file_path: Path to PDF file
        chunk_size: Number of characters per chunk
        chunk_overlap: Overlap between chunks for context preservation
    
    Returns:
        List of Document objects with metadata including page numbers for citations
    """
    if not PDF_AVAILABLE:
        raise ImportError("PDF parsing libraries not installed. Install: pip install pymupdf")
    
    documents = []
    file_name = os.path.basename(file_path)
    
    try:
        if PDF_LIBRARY == "pdfplumber":
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text and text.strip():
                        # Chunk the page text
                        chunks = _chunk_text(text, chunk_size, chunk_overlap)
                        for chunk_idx, chunk in enumerate(chunks):
                            documents.append(Document(
                                page_content=chunk,
                                metadata={
                                    "source": file_name,
                                    "file_path": file_path,
                                    "page_number": page_num,
                                    "chunk_index": chunk_idx,
                                    "total_chunks_in_page": len(chunks),
                                    "file_type": "pdf",
                                    "ingestion_timestamp": datetime.now().isoformat(),
                                    "total_pages": total_pages,
                                    "citation": f"{file_name}, Page {page_num}"
                                }
                            ))
        else:
            # Use PyMuPDF (fitz)
            doc = fitz.open(file_path)
            total_pages = len(doc)
            for page_num in range(total_pages):
                page = doc[page_num]
                text = page.get_text()
                
                if text and text.strip():
                    # Chunk the page text
                    chunks = _chunk_text(text, chunk_size, chunk_overlap)
                    for chunk_idx, chunk in enumerate(chunks):
                        documents.append(Document(
                            page_content=chunk,
                            metadata={
                                "source": file_name,
                                "file_path": file_path,
                                "page_number": page_num + 1,
                                "chunk_index": chunk_idx,
                                "total_chunks_in_page": len(chunks),
                                "file_type": "pdf",
                                "ingestion_timestamp": datetime.now().isoformat(),
                                "total_pages": total_pages,
                                "citation": f"{file_name}, Page {page_num + 1}"
                            }
                        ))
            doc.close()
        
        logger.info(f"✓ Parsed PDF: {file_name}, {len(documents)} chunks from {documents[0].metadata.get('total_pages', 'unknown')} pages")
        return documents
        
    except Exception as e:
        logger.error(f"✗ Error parsing PDF {file_path}: {str(e)}")
        raise


def parse_docx(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Parse DOCX file with paragraph-level tracking and smart chunking.
    
    Args:
        file_path: Path to DOCX file
        chunk_size: Number of characters per chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of Document objects with metadata including paragraph numbers for citations
    """
    if not DOCX_AVAILABLE:
        raise ImportError("DOCX parsing not available. Install: pip install python-docx")
    
    documents = []
    file_name = os.path.basename(file_path)
    
    try:
        doc = DocxDocument(file_path)
        
        # Extract text with paragraph tracking
        paragraphs = []
        for para_idx, para in enumerate(doc.paragraphs):
            text = para.text.strip()
            if text:
                paragraphs.append({
                    "text": text,
                    "paragraph_index": para_idx + 1
                })
        
        if not paragraphs:
            logger.warning(f"No text content found in DOCX: {file_path}")
            return []
        
        # Combine paragraphs and chunk
        full_text = "\n\n".join([p["text"] for p in paragraphs])
        chunks = _chunk_text(full_text, chunk_size, chunk_overlap)
        
        for chunk_idx, chunk in enumerate(chunks):
            # Determine which paragraphs are in this chunk
            start_para = _find_paragraph_for_chunk(chunk, paragraphs, chunk_idx, len(chunks))
            
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "source": file_name,
                    "file_path": file_path,
                    "paragraph_start": start_para,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "file_type": "docx",
                    "ingestion_timestamp": datetime.now().isoformat(),
                    "total_paragraphs": len(paragraphs),
                    "citation": f"{file_name}, Paragraph ~{start_para}"
                }
            ))
        
        logger.info(f"✓ Parsed DOCX: {file_name}, {len(documents)} chunks from {len(paragraphs)} paragraphs")
        return documents
        
    except Exception as e:
        logger.error(f"✗ Error parsing DOCX {file_path}: {str(e)}")
        raise


def parse_txt(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Parse TXT file with encoding detection and smart chunking.
    
    Args:
        file_path: Path to TXT file
        chunk_size: Number of characters per chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of Document objects with metadata including line numbers for citations
    """
    documents = []
    file_name = os.path.basename(file_path)
    
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        text = None
        used_encoding = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                used_encoding = encoding
                break
            except UnicodeDecodeError:
                continue
        
        if text is None:
            # Fallback to binary read
            with open(file_path, 'rb') as f:
                text = f.read().decode('utf-8', errors='ignore')
            used_encoding = 'utf-8 (with errors ignored)'
        
        if not text or not text.strip():
            logger.warning(f"Empty or unreadable TXT file: {file_path}")
            return []
        
        # Split into lines for better chunking
        lines = text.split('\n')
        chunks = _chunk_text(text, chunk_size, chunk_overlap)
        
        for chunk_idx, chunk in enumerate(chunks):
            # Estimate line range for this chunk
            start_line = _estimate_line_number(chunk, lines, chunk_idx, len(chunks))
            
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "source": file_name,
                    "file_path": file_path,
                    "line_start": start_line,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "file_type": "txt",
                    "encoding": used_encoding,
                    "ingestion_timestamp": datetime.now().isoformat(),
                    "total_lines": len(lines),
                    "citation": f"{file_name}, Line ~{start_line}"
                }
            ))
        
        logger.info(f"✓ Parsed TXT: {file_name}, {len(documents)} chunks from {len(lines)} lines (encoding: {used_encoding})")
        return documents
        
    except Exception as e:
        logger.error(f"✗ Error parsing TXT {file_path}: {str(e)}")
        raise


def parse_excel(file_path: str) -> List[Document]:
    """
    Parse Excel file (XLS/XLSX) - enhanced version with better metadata.
    Each row becomes a document with full citation information.
    """
    try:
        df = pd.read_excel(file_path)
        documents = []
        file_name = os.path.basename(file_path)

        for idx, row in df.iterrows():
            content_lines = []

            for col in df.columns:
                value = row[col]
                if pd.notna(value):
                    col_clean = str(col).strip().capitalize()
                    value_clean = str(value).strip()
                    content_lines.append(f"{col_clean}: {value_clean}")

            if content_lines:
                content = "\n".join(content_lines).lower()

                documents.append(Document(
                    page_content=content,
                    metadata={
                        "row_index": idx + 2,  # +2 because Excel is 1-indexed and has header
                        "source": file_name,
                        "file_path": file_path,
                        "columns": list(df.columns),
                        "file_type": "excel",
                        "ingestion_timestamp": datetime.now().isoformat(),
                        "total_rows": len(df),
                        "citation": f"{file_name}, Row {idx + 2}"
                    }
                ))
        
        logger.info(f"✓ Parsed Excel: {file_name}, {len(documents)} rows")
        return documents
        
    except Exception as e:
        logger.error(f"✗ Error parsing Excel {file_path}: {str(e)}")
        raise


def parse_document(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Universal document parser that automatically detects file type and parses accordingly.
    
    Supported formats: PDF, DOCX, DOC, TXT, XLSX, XLS
    
    Args:
        file_path: Path to document file
        chunk_size: Number of characters per chunk (for text-based formats)
        chunk_overlap: Overlap between chunks for context preservation
    
    Returns:
        List of Document objects with comprehensive metadata for citations
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.pdf':
        if not PDF_AVAILABLE:
            raise ImportError("PDF parsing not available. Install: pip install pymupdf")
        return parse_pdf(file_path, chunk_size, chunk_overlap)
    
    elif file_ext in ['.docx', '.doc']:
        if not DOCX_AVAILABLE:
            raise ImportError("DOCX parsing not available. Install: pip install python-docx")
        return parse_docx(file_path, chunk_size, chunk_overlap)
    
    elif file_ext in ['.txt', '.text']:
        return parse_txt(file_path, chunk_size, chunk_overlap)
    
    elif file_ext in ['.xlsx', '.xls']:
        return parse_excel(file_path)
    
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported: .pdf, .docx, .doc, .txt, .xlsx, .xls")


def get_supported_formats() -> List[str]:
    """Return list of supported file formats."""
    formats = ['.xlsx', '.xls', '.txt']
    
    if PDF_AVAILABLE:
        formats.append('.pdf')
    
    if DOCX_AVAILABLE:
        formats.extend(['.docx', '.doc'])
    
    return formats


def check_format_support() -> Dict[str, bool]:
    """Check which formats are currently supported based on installed libraries."""
    return {
        "pdf": PDF_AVAILABLE,
        "docx": DOCX_AVAILABLE,
        "txt": True,
        "excel": True
    }


# Helper functions

def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Split text into chunks with overlap for context preservation.
    Uses sentence boundaries when possible for better semantic coherence.
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        
        # Try to break at sentence boundary (., !, ?)
        sentence_end = max(
            text.rfind('.', start, end),
            text.rfind('!', start, end),
            text.rfind('?', start, end)
        )
        
        if sentence_end > start:
            end = sentence_end + 1
        else:
            # Try to break at paragraph boundary
            para_end = text.rfind('\n\n', start, end)
            if para_end > start:
                end = para_end + 2
            else:
                # Try to break at word boundary
                word_end = text.rfind(' ', start, end)
                if word_end > start:
                    end = word_end
        
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)
        
        start = end - chunk_overlap if chunk_overlap > 0 else end
    
    return chunks


def _find_paragraph_for_chunk(chunk: str, paragraphs: List[Dict], chunk_idx: int, total_chunks: int) -> int:
    """Estimate which paragraph a chunk starts from."""
    if not paragraphs:
        return 1
    
    # Simple estimation based on chunk position
    estimated_para = int((chunk_idx / total_chunks) * len(paragraphs)) + 1
    return min(estimated_para, len(paragraphs))


def _estimate_line_number(chunk: str, lines: List[str], chunk_idx: int, total_chunks: int) -> int:
    """Estimate which line a chunk starts from."""
    if not lines:
        return 1
    
    estimated_line = int((chunk_idx / total_chunks) * len(lines)) + 1
    return min(estimated_line, len(lines))

"""
OCR Service for extracting text from scanned PDFs using Microsoft TrOCR.
Supports both printed and handwritten text recognition with transformer-based models.
Fully offline operation with HuggingFace transformers.
"""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import io

logger = logging.getLogger(__name__)

# TrOCR (Transformer-based OCR)
TROCR_AVAILABLE = False
_trocr_processor = None
_trocr_model = None
_trocr_model_type = None  # 'printed' or 'handwritten'

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image
    TROCR_AVAILABLE = True
    logger.info("TrOCR transformers available")
except ImportError:
    TROCR_AVAILABLE = False
    logger.warning("TrOCR not available. Install: pip install transformers torch pillow")

# PDF processing for OCR
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available for OCR processing")


def initialize_trocr(model_type: str = "printed") -> bool:
    """
    Initialize TrOCR model (lazy loading).
    Model is loaded only once and cached for reuse.
    
    Args:
        model_type: "printed" or "handwritten" (default: "printed")
    
    Returns:
        True if model loaded successfully
    """
    global _trocr_processor, _trocr_model, _trocr_model_type
    
    if not TROCR_AVAILABLE:
        logger.error("TrOCR transformers not available. Install: pip install transformers torch pillow")
        return False
    
    # Use cached model if already loaded with same type
    if _trocr_model is not None and _trocr_model_type == model_type:
        return True
    
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        import torch
        
        logger.info(f"Loading TrOCR {model_type} model (first time only)...")
        
        # Select model based on type
        if model_type == "handwritten":
            model_name = "microsoft/trocr-base-handwritten"
        else:  # printed (default)
            model_name = "microsoft/trocr-base-printed"
        
        # Load processor and model
        _trocr_processor = TrOCRProcessor.from_pretrained(model_name)
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        # Set to eval mode and CPU for memory efficiency
        _trocr_model.eval()
        if torch.cuda.is_available():
            _trocr_model = _trocr_model.to('cpu')  # Force CPU to save GPU memory
        else:
            _trocr_model = _trocr_model.to('cpu')
        
        _trocr_model_type = model_type
        logger.info(f"✓ TrOCR {model_type} model loaded successfully: {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load TrOCR model: {str(e)}")
        _trocr_processor = None
        _trocr_model = None
        _trocr_model_type = None
        return False


def is_trocr_available() -> bool:
    """
    Check if TrOCR is available and can be used.
    
    Returns:
        True if TrOCR is available
    """
    if not TROCR_AVAILABLE:
        return False
    
    # Try to initialize if not already loaded
    if _trocr_model is None:
        return initialize_trocr("printed")
    
    return _trocr_model is not None


def preprocess_image_for_trocr(image: Image.Image) -> Image.Image:
    """
    Preprocess image for TrOCR (convert to RGB, resize if needed).
    
    Args:
        image: PIL Image object
    
    Returns:
        Preprocessed PIL Image (RGB format)
    """
    # TrOCR requires RGB images
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # TrOCR works best with images that are not too large
    # Resize if image is very large (max 384px height recommended)
    max_height = 384
    if image.height > max_height:
        ratio = max_height / image.height
        new_width = int(image.width * ratio)
        new_height = max_height
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        logger.debug(f"Resized image from {image.size} to ({new_width}, {new_height})")
    
    return image


def extract_text_with_trocr(image_data: bytes, 
                            model_type: str = "printed",
                            preprocess: bool = True) -> Dict[str, Any]:
    """
    Extract text from an image using TrOCR (Transformer-based OCR).
    
    Args:
        image_data: Raw image bytes
        model_type: "printed" or "handwritten" (default: "printed")
        preprocess: Whether to preprocess image (default: True)
    
    Returns:
        Dictionary with OCR results:
        {
            "text": extracted_text,
            "confidence": confidence_score (estimated),
            "success": bool,
            "model_type": str
        }
    """
    if not TROCR_AVAILABLE:
        return {
            "text": "",
            "confidence": 0.0,
            "success": False,
            "error": "TrOCR not available. Install: pip install transformers torch pillow",
            "model_type": None
        }
    
    if not initialize_trocr(model_type):
        return {
            "text": "",
            "confidence": 0.0,
            "success": False,
            "error": "Failed to load TrOCR model",
            "model_type": None
        }
    
    try:
        import torch
        
        # Load image
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess if requested
        if preprocess:
            image = preprocess_image_for_trocr(image)
        
        # Process image with TrOCR
        pixel_values = _trocr_processor(images=image, return_tensors="pt").pixel_values
        
        # Generate text
        with torch.no_grad():
            generated_ids = _trocr_model.generate(pixel_values)
            generated_text = _trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # TrOCR doesn't provide confidence scores directly
        # Estimate confidence based on text length and model type
        # (longer extracted text generally indicates better recognition)
        text_length = len(generated_text.strip())
        estimated_confidence = min(95.0, 70.0 + (text_length / 10)) if text_length > 0 else 0.0
        
        return {
            "text": generated_text.strip(),
            "confidence": round(estimated_confidence, 2),
            "success": True,
            "char_count": text_length,
            "model_type": model_type
        }
        
    except Exception as e:
        logger.error(f"TrOCR extraction failed: {str(e)}")
        return {
            "text": "",
            "confidence": 0.0,
            "success": False,
            "error": str(e),
            "model_type": model_type
        }


def detect_text_type(image_data: bytes) -> str:
    """
    Simple heuristic to detect if text is handwritten or printed.
    This is a basic implementation - can be enhanced with a classifier.
    
    Args:
        image_data: Raw image bytes
    
    Returns:
        "handwritten" or "printed"
    """
    # For now, default to "printed" for most cases
    # Can be enhanced with a proper classifier model
    # Users can also manually specify via API
    return "printed"


def ocr_pdf_page(page, page_num: int, model_type: str = "auto") -> Dict[str, Any]:
    """
    Extract text from a PDF page using TrOCR.
    
    Args:
        page: PyMuPDF page object
        page_num: Page number (1-indexed)
        model_type: "printed", "handwritten", or "auto" (default: "auto")
    
    Returns:
        Dictionary with OCR results for the page
    """
    try:
        # Render page to image at higher DPI for better OCR
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
        img_data = pix.tobytes("png")
        
        # Auto-detect text type if needed
        if model_type == "auto":
            model_type = detect_text_type(img_data)
        
        # Run TrOCR
        ocr_result = extract_text_with_trocr(img_data, model_type=model_type)
        
        return {
            "page_number": page_num,
            "text": ocr_result.get("text", ""),
            "confidence": ocr_result.get("confidence", 0),
            "success": ocr_result.get("success", False),
            "char_count": ocr_result.get("char_count", 0),
            "model_type": ocr_result.get("model_type", model_type),
            "error": ocr_result.get("error")
        }
        
    except Exception as e:
        logger.error(f"Error OCR'ing page {page_num} with TrOCR: {str(e)}")
        return {
            "page_number": page_num,
            "text": "",
            "confidence": 0,
            "success": False,
            "error": str(e),
            "model_type": model_type
        }


def ocr_pdf_document(pdf_path: str, 
                     model_type: str = "auto",
                     max_pages: Optional[int] = None) -> Dict[str, Any]:
    """
    Extract text from entire PDF using TrOCR.
    
    Args:
        pdf_path: Path to PDF file
        model_type: "printed", "handwritten", or "auto" (default: "auto")
        max_pages: Maximum pages to process (None = all pages)
    
    Returns:
        Dictionary with OCR results:
        {
            "success": bool,
            "pages": List[page_results],
            "total_pages": int,
            "processed_pages": int,
            "avg_confidence": float,
            "total_text_length": int,
            "ocr_engine": "TrOCR"
        }
    """
    if not PYMUPDF_AVAILABLE:
        return {
            "success": False,
            "error": "PyMuPDF not available",
            "ocr_engine": "TrOCR"
        }
    
    if not is_trocr_available():
        return {
            "success": False,
            "error": "TrOCR not available. Install: pip install transformers torch pillow",
            "ocr_engine": "TrOCR"
        }
    
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        pages_to_process = min(total_pages, max_pages) if max_pages else total_pages
        
        logger.info(f"Starting TrOCR on {pdf_path}: {pages_to_process} pages (model_type: {model_type})")
        
        page_results = []
        total_confidence = 0
        total_text_length = 0
        successful_pages = 0
        
        for page_num in range(pages_to_process):
            logger.info(f"TrOCR processing page {page_num + 1}/{pages_to_process}...")
            page = doc[page_num]
            
            result = ocr_pdf_page(page, page_num + 1, model_type)
            page_results.append(result)
            
            if result["success"]:
                successful_pages += 1
                total_confidence += result["confidence"]
                total_text_length += result["char_count"]
        
        doc.close()
        
        avg_confidence = total_confidence / successful_pages if successful_pages > 0 else 0
        
        return {
            "success": True,
            "pages": page_results,
            "total_pages": total_pages,
            "processed_pages": pages_to_process,
            "successful_pages": successful_pages,
            "avg_confidence": round(avg_confidence, 2),
            "total_text_length": total_text_length,
            "file_name": os.path.basename(pdf_path),
            "ocr_engine": "TrOCR",
            "model_type": model_type
        }
        
    except Exception as e:
        logger.error(f"Error during PDF OCR with TrOCR: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "ocr_engine": "TrOCR"
        }


def detect_scanned_pdf(pdf_path: str, text_threshold: float = 0.1) -> Tuple[bool, Dict[str, Any]]:
    """
    Detect if a PDF is scanned (image-only) or has extractable text.
    
    Args:
        pdf_path: Path to PDF file
        text_threshold: Minimum ratio of pages with text to consider PDF as native (default: 0.1)
    
    Returns:
        Tuple of (is_scanned, detection_info)
        - is_scanned: True if PDF is scanned/image-only
        - detection_info: Dictionary with detection statistics
    """
    if not PYMUPDF_AVAILABLE:
        logger.error("PyMuPDF required for PDF type detection")
        return False, {"error": "PyMuPDF not available"}
    
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        pages_with_text = 0
        pages_with_images = 0
        total_text_length = 0
        
        for page_num in range(min(total_pages, 10)):  # Sample first 10 pages for speed
            page = doc[page_num]
            
            # Check for text
            text = page.get_text().strip()
            if text and len(text) > 50:  # More than 50 chars = has text
                pages_with_text += 1
                total_text_length += len(text)
            
            # Check for images
            images = page.get_images(full=True)
            if images:
                pages_with_images += 1
        
        doc.close()
        
        # Calculate metrics
        sampled_pages = min(total_pages, 10)
        text_page_ratio = pages_with_text / sampled_pages if sampled_pages > 0 else 0
        avg_text_per_page = total_text_length / sampled_pages if sampled_pages > 0 else 0
        
        # Decision logic
        is_scanned = text_page_ratio < text_threshold and pages_with_images > 0
        
        detection_info = {
            "total_pages": total_pages,
            "sampled_pages": sampled_pages,
            "pages_with_text": pages_with_text,
            "pages_with_images": pages_with_images,
            "text_page_ratio": round(text_page_ratio, 2),
            "avg_text_per_page": round(avg_text_per_page, 2),
            "is_scanned": is_scanned,
            "confidence": "high" if text_page_ratio < 0.05 or text_page_ratio > 0.9 else "medium"
        }
        
        logger.info(f"PDF Detection: {'SCANNED' if is_scanned else 'NATIVE'} - {detection_info}")
        return is_scanned, detection_info
        
    except Exception as e:
        logger.error(f"Error detecting PDF type: {str(e)}")
        return False, {"error": str(e)}


def get_ocr_status() -> Dict[str, Any]:
    """
    Get OCR system status and capabilities.
    
    Returns:
        Dictionary with OCR status information
    """
    trocr_available = is_trocr_available()
    model_loaded = _trocr_model is not None
    
    status = {
        "trocr_available": trocr_available,
        "trocr_model_loaded": model_loaded,
        "trocr_model_type": _trocr_model_type if model_loaded else None,
        "pymupdf_available": PYMUPDF_AVAILABLE,
        "ocr_engine": "TrOCR",
        "supported_models": ["printed", "handwritten", "auto"],
        "offline_capable": True,
        "handwritten_support": True
    }
    
    if model_loaded:
        status["current_model"] = f"microsoft/trocr-base-{_trocr_model_type}"
    
    return status


# Legacy compatibility functions (for backward compatibility)
def is_tesseract_installed() -> bool:
    """
    Legacy function - always returns False as Tesseract is replaced by TrOCR.
    Kept for backward compatibility.
    """
    return False


def extract_text_with_ocr(image_data: bytes, 
                          language: str = 'eng',
                          preprocess: bool = True,
                          config: str = '--psm 1') -> Dict[str, Any]:
    """
    Legacy function - redirects to TrOCR.
    Kept for backward compatibility.
    """
    logger.info("Using TrOCR instead of Tesseract (legacy function call)")
    return extract_text_with_trocr(image_data, model_type="printed", preprocess=preprocess)

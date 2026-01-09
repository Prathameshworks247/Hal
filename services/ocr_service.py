"""
OCR Service for extracting text from scanned PDFs using Tesseract OCR.
Supports automatic detection of scanned PDFs and text extraction with confidence scoring.
"""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import io

logger = logging.getLogger(__name__)

# Tesseract OCR
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
    logger.info("Tesseract OCR is available")
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("Tesseract OCR not available. Install: pip install pytesseract pillow")

# PDF processing for OCR
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available for OCR processing")

# Image preprocessing - DISABLED due to numpy/cv2 segfault on macOS
# To enable, fix numpy/OpenCV compatibility: pip install --upgrade numpy opencv-python-headless
CV2_AVAILABLE = False
logger.warning("OpenCV disabled due to compatibility issues. OCR preprocessing will be limited.")


def is_tesseract_installed() -> bool:
    """
    Check if Tesseract OCR is installed on the system.
    
    Returns:
        True if Tesseract is installed and accessible
    """
    if not TESSERACT_AVAILABLE:
        return False
    
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception as e:
        logger.warning(f"Tesseract not found in system: {str(e)}")
        return False


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


def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    Preprocess image to improve OCR accuracy.
    
    Args:
        image: PIL Image object
    
    Returns:
        Preprocessed PIL Image
    """
    if not CV2_AVAILABLE:
        # Basic preprocessing without OpenCV
        return image.convert('L')  # Convert to grayscale
    
    try:
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply thresholding to get better contrast
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Noise removal
        gray = cv2.medianBlur(gray, 3)
        
        # Convert back to PIL
        return Image.fromarray(gray)
        
    except Exception as e:
        logger.warning(f"Image preprocessing failed: {str(e)}, using original")
        return image.convert('L')


def extract_text_with_ocr(image_data: bytes, 
                          language: str = 'eng',
                          preprocess: bool = True,
                          config: str = '--psm 1') -> Dict[str, Any]:
    """
    Extract text from an image using Tesseract OCR.
    
    Args:
        image_data: Raw image bytes
        language: Tesseract language code (default: 'eng')
        preprocess: Whether to preprocess image (default: True)
        config: Tesseract configuration string
    
    Returns:
        Dictionary with OCR results:
        {
            "text": extracted_text,
            "confidence": confidence_score,
            "success": bool
        }
    """
    if not TESSERACT_AVAILABLE:
        return {
            "text": "",
            "confidence": 0.0,
            "success": False,
            "error": "Tesseract not available"
        }
    
    if not is_tesseract_installed():
        return {
            "text": "",
            "confidence": 0.0,
            "success": False,
            "error": "Tesseract not installed on system"
        }
    
    try:
        # Load image
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess if requested
        if preprocess:
            image = preprocess_image_for_ocr(image)
        
        # Run OCR
        text = pytesseract.image_to_string(image, lang=language, config=config)
        
        # Get confidence data
        try:
            data = pytesseract.image_to_data(image, lang=language, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        except:
            avg_confidence = 0
        
        return {
            "text": text.strip(),
            "confidence": round(avg_confidence, 2),
            "success": True,
            "char_count": len(text.strip())
        }
        
    except Exception as e:
        logger.error(f"OCR extraction failed: {str(e)}")
        return {
            "text": "",
            "confidence": 0.0,
            "success": False,
            "error": str(e)
        }


def ocr_pdf_page(page, page_num: int, language: str = 'eng') -> Dict[str, Any]:
    """
    Extract text from a PDF page using OCR.
    
    Args:
        page: PyMuPDF page object
        page_num: Page number (1-indexed)
        language: Tesseract language code
    
    Returns:
        Dictionary with OCR results for the page
    """
    try:
        # Render page to image at higher DPI for better OCR
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
        img_data = pix.tobytes("png")
        
        # Run OCR
        ocr_result = extract_text_with_ocr(img_data, language=language)
        
        return {
            "page_number": page_num,
            "text": ocr_result.get("text", ""),
            "confidence": ocr_result.get("confidence", 0),
            "success": ocr_result.get("success", False),
            "char_count": len(ocr_result.get("text", "")),
            "error": ocr_result.get("error")
        }
        
    except Exception as e:
        logger.error(f"Error OCR'ing page {page_num}: {str(e)}")
        return {
            "page_number": page_num,
            "text": "",
            "confidence": 0,
            "success": False,
            "error": str(e)
        }


def ocr_pdf_document(pdf_path: str, 
                     language: str = 'eng',
                     max_pages: Optional[int] = None) -> Dict[str, Any]:
    """
    Extract text from entire PDF using OCR.
    
    Args:
        pdf_path: Path to PDF file
        language: Tesseract language code (default: 'eng')
        max_pages: Maximum pages to process (None = all pages)
    
    Returns:
        Dictionary with OCR results:
        {
            "success": bool,
            "pages": List[page_results],
            "total_pages": int,
            "processed_pages": int,
            "avg_confidence": float,
            "total_text_length": int
        }
    """
    if not PYMUPDF_AVAILABLE:
        return {
            "success": False,
            "error": "PyMuPDF not available"
        }
    
    if not is_tesseract_installed():
        return {
            "success": False,
            "error": "Tesseract not installed on system. Install with: apt-get install tesseract-ocr (Linux) or brew install tesseract (Mac)"
        }
    
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        pages_to_process = min(total_pages, max_pages) if max_pages else total_pages
        
        logger.info(f"Starting OCR on {pdf_path}: {pages_to_process} pages")
        
        page_results = []
        total_confidence = 0
        total_text_length = 0
        successful_pages = 0
        
        for page_num in range(pages_to_process):
            logger.info(f"OCR processing page {page_num + 1}/{pages_to_process}...")
            page = doc[page_num]
            
            result = ocr_pdf_page(page, page_num + 1, language)
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
            "ocr_engine": "Tesseract"
        }
        
    except Exception as e:
        logger.error(f"Error during PDF OCR: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def get_ocr_status() -> Dict[str, Any]:
    """
    Get OCR system status and capabilities.
    
    Returns:
        Dictionary with OCR status information
    """
    return {
        "tesseract_available": TESSERACT_AVAILABLE,
        "tesseract_installed": is_tesseract_installed(),
        "pymupdf_available": PYMUPDF_AVAILABLE,
        "opencv_available": CV2_AVAILABLE,
        "preprocessing_available": CV2_AVAILABLE,
        "tesseract_version": pytesseract.get_tesseract_version() if is_tesseract_installed() else None,
        "supported_languages": ["eng", "hin", "fra", "deu", "spa"] if is_tesseract_installed() else [],
        "ocr_engine": "Tesseract"
    }

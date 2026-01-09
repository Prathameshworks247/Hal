# app.py
from datetime import datetime
import os
import logging
import io

# NOTE: CV2 import disabled due to numpy/OpenCV compatibility issues on macOS
# If you need OCR preprocessing, ensure compatible versions are installed:
# pip install --upgrade numpy opencv-python-headless
CV2_AVAILABLE = False
# try:
#     import cv2
#     import numpy as np
#     CV2_AVAILABLE = True
# except ImportError:
#     CV2_AVAILABLE = False
#     logging.warning("OpenCV (cv2) not available. Some features may be limited.")
import uuid
import shutil
from collections import defaultdict
import pandas as pd
import re
from fastapi.responses import StreamingResponse,JSONResponse
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Any, Optional
from fastapi.staticfiles import StaticFiles
from collections import defaultdict
from services.llm import get_llm
from fastapi.encoders import jsonable_encoder
from functools import lru_cache
import shutil
from services.llm import get_llm
from models.models import ExcelFileInput,GetRows,NamesReq,QueryRequestFile,ShapeDetectionResponse,QueryRequest
from services.chain_service import get_chain, get_chain_file, verify
from services.prompt_verifier import verify_prompt
from services.citation_service import create_traceable_response, extract_citations_from_sources
from services.document_parser import parse_document, get_supported_formats, check_format_support
from services.pdf_citation_service import get_citations_for_session, get_citation_by_id
from fastapi.responses import FileResponse
from services.incremental_learning import IncrementalLearningManager, create_or_update_index
from services.rbac_service import get_rbac_manager, Role, Permission
from services.ingest import ingest_single_file, ingest_directory, rebuild_index_from_scratch, get_index_info
from services.similarity_service import  get_similar_records_with_metadata
from services.parsers import process_snag_query_json, process_file_query_json, display_results_as_json
from utils.utils import test_retriever, convert_numpy
from services.chain_service import get_analytics_chain
from services.parsers import process_snag_query_json_analysis
from services.chain_service import get_analytics_chain_from_xls
from services.similarity_service import has_semantic_meaning
# DISABLED: vision.funcs uses cv2 which causes segfault on macOS
# from vision.funcs import detect_shapes,get_pixels_per_mm,save_to_csv

app = FastAPI(title="Aircraft AI API", description="")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend's origin (e.g., React dev server)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-CSV-Download-URL", "X-Photo-Download-URL", "X-Shapes-Detected"]  # Key addition: exposes them globally
) 

app.mount("/static", StaticFiles(directory="static"), name="static")
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/test-headers")
def test_headers():
    headers = {
        "X-CSV-Download-URL": "/static/test.csv",
        "X-Photo-Download-URL": "/static/test.jpg",
        "X-Shapes-Detected": "5",
        "Access-Control-Expose-Headers": "X-CSV-Download-URL,X-Photo-Download-URL,X-Shapes-Detected"
    }
    return JSONResponse(content={"message": "Test"}, headers=headers)

@app.post("/detect-shapes", response_model=ShapeDetectionResponse)
async def detect_shapes_endpoint(
    file: UploadFile = File(...),
    aruco_marker_size: float = Form(50.0, description="ArUco marker size in mm")
):
    """
    Upload an image with shapes and ArUco marker to get measurements
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        unique_id = str(uuid.uuid4())
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        pixels_per_mm = get_pixels_per_mm(image, aruco_marker_size)
        
        if pixels_per_mm is None:
            return ShapeDetectionResponse(
                success=False,
                message="ArUco marker not detected in the image. Please ensure a clear ArUco marker is present.",
                shapes_detected=0
            )
        
        # Detect shapes
        output_image, shape_data = detect_shapes(image, pixels_per_mm)
        
        # Save processed image
        output_image_path = f"static/processed_{unique_id}.jpg"
        cv2.imwrite(output_image_path, output_image)

        # Save CSV data
        csv_path = f"static/shapes_{unique_id}.csv"
        save_to_csv(shape_data, csv_path)

        # Convert image to memory buffer
        _, img_encoded = cv2.imencode('.jpg', output_image)
        img_bytes = io.BytesIO(img_encoded.tobytes())

        # Convert numpy types for JSON headers
        for shape in shape_data:
            for key, value in shape.items():
                if isinstance(value, (np.generic, np.ndarray)):
                    shape[key] = value.item() if hasattr(value, "item") else value.tolist()

        # Create downloadable CSV link
        csv_url = f"static/shapes_{unique_id}.csv"
        photo_url = f'static/processed_{unique_id}.jpg'

        headers = {
            "X-CSV-Download-URL": csv_url,
            "X-Photo-Download-URL":photo_url,
            "X-Shapes-Detected": str(len(shape_data))
        }
        return StreamingResponse(
            content=img_bytes,
            media_type="image/jpeg",
            headers=headers
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/system/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Shape Detection API"}


@app.delete("/user/end-session/{session_id}")
async def end_session(session_id: str):
    """
    End a session and delete all associated data (FAISS index, files, metadata).
    
    Args:
        session_id: Session identifier to delete
        
    Returns:
        Status of deletion operation
    """
    try:
        from services.session_faiss_manager import SessionFAISSManager
        from services.multimodal_embeddings import get_multimodal_embeddings
        from services.session_storage import session_exists, get_session_size_mb
        
        if not session_exists(session_id):
            return JSONResponse(
                status_code=404,
                content={
                    "status": "not_found",
                    "session_id": session_id,
                    "message": "Session does not exist"
                }
            )
        
        # Get session size before deletion
        size_mb = get_session_size_mb(session_id) or 0
        
        # Initialize manager and destroy session
        embeddings = get_multimodal_embeddings(model_path="./all-MiniLM-L6-v2", device='cpu')
        session_manager = SessionFAISSManager(session_id, embeddings)
        
        # Get metadata before deletion
        metadata = session_manager.get_session_metadata()
        
        success = session_manager.destroy_session()
        
        if success:
            return {
                "status": "deleted",
                "session_id": session_id,
                "deleted_files": 1 if metadata.has_uploaded_file else 0,
                "deleted_embeddings": metadata.total_embeddings,
                "conversation_turns": metadata.conversation_turns,
                "size_mb": size_mb,
                "message": "Session and all associated data deleted successfully"
            }
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "session_id": session_id,
                    "message": "Failed to delete session"
                }
            )
            
    except Exception as e:
        logger.exception(f"Error deleting session {session_id}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )


@app.get("/admin/sessions")
async def list_sessions():
    """
    List all active sessions with metadata.
    Admin endpoint for monitoring session storage.
    
    Returns:
        List of active sessions with details
    """
    try:
        from services.session_storage import get_all_session_info, list_active_sessions
        
        sessions_info = get_all_session_info()
        active_session_ids = list_active_sessions()
        
        # Calculate total storage used
        from services.session_storage import get_session_size_mb
        total_size_mb = sum(get_session_size_mb(sid) or 0 for sid in active_session_ids)
        
        return {
            "status": "success",
            "active_sessions": len(active_session_ids),
            "sessions": [
                {
                    "session_id": info.session_id,
                    "created_at": info.created_at.isoformat(),
                    "last_accessed": info.last_accessed.isoformat(),
                    "age_hours": info.age_hours,
                    "has_uploaded_file": info.has_uploaded_file,
                    "uploaded_file_name": info.uploaded_file_name,
                    "conversation_turns": info.conversation_turns,
                    "total_embeddings": info.total_embeddings
                }
                for info in sessions_info
            ],
            "total_storage_mb": round(total_size_mb, 2)
        }
        
    except Exception as e:
        logger.exception("Error listing sessions")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )


@app.post("/admin/cleanup-sessions")
async def cleanup_sessions(max_age_hours: int = 24):
    """
    Manually trigger cleanup of expired sessions.
    
    Args:
        max_age_hours: Maximum age in hours before deletion (default: 24)
        
    Returns:
        Number of sessions cleaned up
    """
    try:
        from services.session_storage import cleanup_expired_sessions
        
        deleted_count = cleanup_expired_sessions(max_age_hours)
        
        return {
            "status": "success",
            "deleted_sessions": deleted_count,
            "max_age_hours": max_age_hours,
            "message": f"Cleaned up {deleted_count} expired sessions"
        }
        
    except Exception as e:
        logger.exception("Error during session cleanup")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )


@app.on_event("startup")
async def cleanup_old_files():
    """Clean up old files and expired sessions on server startup"""
    try:
        # Cleanup old static files
        for file in os.listdir("static"):
            if file.startswith(("processed_", "shapes_")):
                os.remove(f"static/{file}")
    except:
        pass
    
    # Cleanup expired sessions (older than 24 hours)
    try:
        from services.session_storage import cleanup_expired_sessions
        deleted = cleanup_expired_sessions(max_age_hours=24)
        if deleted > 0:
            logger.info(f"Startup cleanup: Deleted {deleted} expired sessions")
    except Exception as e:
        logger.error(f"Error during startup session cleanup: {str(e)}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@lru_cache()
def get_chain_cached(department: str = None, query: str = None):
    # Normalize None to empty string for caching
    dept_key = department if department is not None else ""
    query_key = query if query is not None else ""
    return get_chain(department=dept_key if dept_key else None, query=query_key if query_key else None)
def get_chain_file_chached(file_name, pb_number):
    return get_chain_file(file_name, pb_number)


@app.post("/user/rectify")
async def rectification(request: QueryRequestFile, background_tasks: BackgroundTasks) -> Dict[Any, Any]:
    try:
        file_name = request.file_name
        pb_number = request.pb_number or "default"  # Use default if not provided
        final_query = request.query
        conversation_history = request.conversation_history or []
        session_id = request.session_id
        
        logger.info(f"📥 Received query request:")
        logger.info(f"   Query: {final_query[:50]}...")
        logger.info(f"   File: {file_name}")
        logger.info(f"   Session: {session_id or 'NEW'}")
        logger.info(f"   History: {len(conversation_history)} messages")
        
        # Get or create session_id
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.info(f"Created new session: {session_id}")
        else:
            logger.info(f"Using existing session: {session_id}")
        
        # Initialize SessionFAISSManager
        from services.session_faiss_manager import SessionFAISSManager
        from services.multimodal_embeddings import get_multimodal_embeddings
        
        embeddings = get_multimodal_embeddings(model_path="./all-MiniLM-L6-v2", device='cpu')
        session_manager = SessionFAISSManager(session_id, embeddings)
        
        # Process conversation context
        from services.conversation_service import process_conversational_query
        
        conversation_context = process_conversational_query(
            final_query,
            [{"role": msg.role, "content": msg.content} for msg in conversation_history] if conversation_history else None
        )
        
        # Use contextualized query for RAG retrieval
        if conversation_context["has_context"]:
            logger.info(f"Using conversational context: {conversation_context['context_summary']}")
            # Use standalone query for better retrieval
            search_query = conversation_context["standalone_query"]
        else:
            search_query = final_query
        
        # Enhanced prompt verification
        is_valid, error_msg, verification_details = verify_prompt(final_query, context="aircraft")
        if not is_valid:
            logger.warning(f"Query verification failed: {error_msg}")
            return {
                "error": error_msg,
                "verification_details": verification_details
            }
        
        logger.info(f"Query verified successfully. Quality score: {verification_details.get('quality_score', 0):.2f}")
        
        # Optional: Extract snag if present (for backward compatibility)
        # But don't require it - allow general queries too
        match = re.search(r"Snag:\s*(.*?)(\s+\w+:|$)", final_query)
        if match:
            snag_text = match.group(1).strip()
            # Additional semantic check on snag text
            if not has_semantic_meaning(snag_text):
                logger.warning(f"Snag text lacks semantic meaning: {snag_text}")
                # Don't block - just log warning

        print("🚁 Aircraft Snag Resolution System - JSON Output")
        if file_name == 'default':    
            # Route to department-specific FAISS index
            department = request.department if hasattr(request, 'department') and request.department else None
            logger.info(f"📂 Department routing: {department if department else 'null (will infer)'}")
            
            chain, db = get_chain_cached(department=department or "", query=final_query or "")
            
            if db is None:
                return {
                    "error": "Failed to load FAISS index for department routing",
                    "session_id": session_id
                }
            
            if os.getenv("DEBUG_MODE") == "1":
                test_retriever(db, "hydraulic system pressure low")

            print("🔍 Search Query:\n", search_query)
            print("🔍 User Query:\n", final_query)

            json_results, rectification_text = process_snag_query_json(
                chain, db, search_query, final_query, conversation_context,
                session_manager=session_manager,
                citation_session_id=session_id
            )
            
            # Add conversation memory to SESSION_FAISS (in background)
            if rectification_text:
                background_tasks.add_task(
                    session_manager.add_conversation_memory,
                    final_query,
                    rectification_text
                )
            
            # Add session_id to response
            json_results["session_id"] = session_id

            return jsonable_encoder(convert_numpy(json_results))

        else:
            # File-specific query - USE SESSION FAISS
            print(f"📄 Processing file-specific query for: {file_name}")
            
            # Check if session has uploaded file
            if not session_manager.has_uploaded_file():
                return {
                    "error": f"File '{file_name}' not found in session. Please upload the file first.",
                    "session_id": session_id,
                    "suggestion": "Use file_name='default' to search global knowledge base, or upload a file with this session_id first."
                }
            
            print("🔍 Search Query:\n", search_query)
            print("🔍 User Query:\n", final_query)
            print(f"📂 Querying from SESSION_FAISS (session: {session_id})")
            
            # For SESSION_FAISS queries, we don't need to load the global department-based index
            # process_file_query_json() will handle SESSION_FAISS internally and create its own chain
            # Pass None values that will be replaced by process_file_query_json() when using SESSION_FAISS
            chain = None
            db = None
            
            json_results, rectification_text = process_file_query_json(
                chain, db, search_query, final_query, conversation_context,
                session_manager=session_manager,
                citation_session_id=session_id
            )
            
            # Add conversation memory to SESSION_FAISS (in background)
            if rectification_text:
                background_tasks.add_task(
                    session_manager.add_conversation_memory,
                    final_query,
                    rectification_text
                )
            
            # Add session_id to response
            json_results["session_id"] = session_id
            
            return jsonable_encoder(convert_numpy(json_results))

    except Exception as e:
        logger.exception("Error in rectification endpoint")
        return {"error": str(e)}

# @app.post("/rectify-file")
# async def rectify_file(request: QueryRequestFile) -> Dict[Any, Any]:
#     try:
#         file_name = request.file_name
#         pb_number = request.pb_number
#         final_query = request.query
#         qa_chain, vectorstore = get_chain_file(file_name,pb_number)
        
#         # Get AI-generated rectification
#         response = qa_chain.invoke({"question": final_query})

#         # Extract result
#         if isinstance(response, dict):
#             rectification = response.get('result', response.get('answer', str(response)))
#         else:
#             rectification = str(response)
        
#         similar_snags = get_similar_records_with_metadata(vectorstore, final_query, k=5)
#         json_results = display_results_as_json(rectification, similar_snags, final_query)
#         return jsonable_encoder(convert_numpy(json_results))
#     except Exception as e:
#         logger.exception("Error during rectification")
#         return {"error": str(e)}



@app.post("/user/query-pdf-images")
async def query_pdf_images(request: QueryRequest) -> Dict[Any, Any]:
    """
    Query specifically for images extracted from PDFs in the knowledge base.
    Returns only image descriptions with their source PDF information.
    
    Args:
        query: Text query to search image descriptions
    
    Returns:
        JSON response with matching images and their descriptions
    """
    try:
        query = request.query
        logger.info(f"Querying PDF images with: {query}")
        
        # Get vector store
        chain, db = get_chain_cached()
        
        # Search for similar documents (including images)
        docs_with_scores = db.similarity_search_with_score(query, k=20)
        
        # Filter for only image descriptions
        image_results = []
        for doc, score in docs_with_scores:
            metadata = doc.metadata
            
            # Only include image descriptions
            if metadata.get("type") == "image_description":
                image_results.append({
                    "description": doc.page_content,
                    "similarity_score": float(1 - score),  # Convert distance to similarity
                    "source": {
                        "file": metadata.get("source", "unknown"),
                        "file_path": metadata.get("file_path", ""),
                        "page": metadata.get("page_number", 0),
                        "image_index": metadata.get("image_index", 0),
                        "image_format": metadata.get("image_format", "unknown")
                    },
                    "metadata": {
                        "authoritative": metadata.get("authoritative", False),
                        "confidence": metadata.get("confidence", "unknown"),
                        "citation": metadata.get("citation", "")
                    }
                })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "status": "success",
            "total_images_found": len(image_results),
            "images": image_results[:10]  # Return top 10
        }
        
    except Exception as e:
        logger.exception(f"Error querying PDF images: {str(e)}")
        return {
            "error": str(e),
            "status": "failed",
            "timestamp": datetime.now().isoformat()
        }


@app.get("/user/pdf-images/{filename}")
async def get_pdf_images(filename: str, pb_number: str = "default") -> Dict[Any, Any]:
    """
    Get all images extracted from a specific PDF.
    
    Args:
        filename: PDF filename
        pb_number: User/project identifier
    
    Returns:
        List of all images with descriptions from the PDF
    """
    try:
        logger.info(f"Getting images from PDF: {filename}")
        
        # Get vector store
        chain, db = get_chain_cached()
        
        # Get all documents from the index
        # We'll use a broad search and filter by filename
        all_docs = db.similarity_search("*", k=1000)  # Get many docs
        
        # Filter for images from this specific PDF
        pdf_images = []
        for doc in all_docs:
            metadata = doc.metadata
            
            if (metadata.get("type") == "image_description" and 
                metadata.get("source", "").startswith(filename.split('.')[0])):
                
                pdf_images.append({
                    "description": doc.page_content,
                    "page": metadata.get("page_number", 0),
                    "image_index": metadata.get("image_index", 0),
                    "image_format": metadata.get("image_format", "unknown"),
                    "source": {
                        "file": metadata.get("source", "unknown"),
                        "file_path": metadata.get("file_path", ""),
                        "citation": metadata.get("citation", "")
                    }
                })
        
        # Sort by page and image index
        pdf_images.sort(key=lambda x: (x["page"], x["image_index"]))
        
        return {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "pb_number": pb_number,
            "status": "success",
            "total_images": len(pdf_images),
            "images": pdf_images
        }
        
    except Exception as e:
        logger.exception(f"Error getting PDF images: {str(e)}")
        return {
            "error": str(e),
            "status": "failed",
            "timestamp": datetime.now().isoformat()
        }


@app.post("/user/query-image")
async def query_image(
    image: UploadFile = File(...),
    query: str = Form(...),
    pb_number: str = Form("default")
) -> Dict[Any, Any]:
    """
    Query using an uploaded image. The image is analyzed by BLIP vision model
    and the description is used to search the knowledge base.
    
    Args:
        image: Image file (JPG, PNG, etc.)
        query: Optional text query to combine with image
        pb_number: User/project identifier
    
    Returns:
        JSON response with image description and relevant documents
    """
    try:
        logger.info(f"Received image query: {image.filename}")
        
        # Read image data
        image_data = await image.read()
        
        # Generate image description using BLIP
        from services.document_parser import _generate_image_description
        
        logger.info("Generating image description with BLIP...")
        image_description = _generate_image_description(
            image_data,
            image.filename.split('.')[-1].upper() if '.' in image.filename else "PNG"
        )
        
        logger.info(f"Image description: {image_description}")
        
        # Combine image description with text query
        combined_query = f"{query}\n\nImage content: {image_description}" if query else image_description
        
        # Query the knowledge base
        chain, db = get_chain_cached()
        
        logger.info(f"Querying knowledge base with: {combined_query}")
        json_results = process_snag_query_json(chain, db, combined_query)
        
        # Add image description to response
        json_results["image_analysis"] = {
            "filename": image.filename,
            "description": image_description,
            "combined_query": combined_query
        }
        
        return jsonable_encoder(convert_numpy(json_results))
        
    except Exception as e:
        logger.exception(f"Error processing image query: {str(e)}")
        return {
            "error": str(e),
            "status": "failed",
            "timestamp": datetime.now().isoformat()
        }

    
@app.post("/user/file-columns", response_model=Dict[str, List[str]])
def get_unique_row(request: GetRows):
    try:
        DIR = f"uploaded_excels/{request.pb_number}"
        df = pd.read_excel(f"{DIR}/{request.filename}")
        columns = list(df.columns)
        temp = defaultdict(list)
        dic = defaultdict(list)

        for column in columns:
            unique_vals = df[column].dropna().unique().tolist()
            temp[column] = [str(v) if not isinstance(v, (str, int, float, bool)) else v for v in unique_vals]
        for key, value in temp.items():
            if key.strip().lower() == "rectification" or key.strip().lower() == "snag":
                continue
            elif len(temp[key]) < 50:
                dic[key] = value
            else:
                dic[key] = []
        return JSONResponse(content=dic)
    except Exception as e:
        logger.exception("Error retrieving unique column values")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/user/store-files")
async def store_files(
    files: List[UploadFile] = File(...),
    pb_number: str = Form(...),
    is_scanned: bool = Form(False),
    session_id: Optional[str] = Form(None)
):
    """
    Multi-file upload endpoint.
    Accepts multiple files and embeds all into a single SESSION FAISS index.
    """
    try:
        if not files:
            return JSONResponse(
                status_code=400,
                content={"error": "No files provided"}
            )
        
        # Get or create session_id
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.info(f"Created new session for multi-file upload: {session_id}")
        else:
            logger.info(f"Using existing session for multi-file upload: {session_id}")
        
        UPLOAD_DIR = f"uploaded_excels/{pb_number}"
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # Get supported formats dynamically
        supported_formats = get_supported_formats()
        
        # Process all files
        from services.document_parser import parse_document
        from services.session_faiss_manager import SessionFAISSManager
        from services.multimodal_embeddings import get_multimodal_embeddings
        from services.ocr_service import get_ocr_status, is_tesseract_installed, detect_scanned_pdf
        
        # Initialize SessionFAISSManager once for all files
        embeddings = get_multimodal_embeddings(model_path="./all-MiniLM-L6-v2", device='cpu')
        session_manager = SessionFAISSManager(session_id, embeddings)
        
        processed_files = []
        failed_files = []
        total_chunks = 0
        ocr_statuses = {}
        
        for file in files:
            try:
                file_ext = os.path.splitext(file.filename)[1].lower()
                
                if file_ext not in supported_formats:
                    failed_files.append({
                        "file_name": file.filename,
                        "error": f"Unsupported file format: {file_ext}"
                    })
                    continue
                
                # Save file
                file_name = f"{os.path.splitext(file.filename)[0]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}{file_ext}"
                file_location = os.path.join(UPLOAD_DIR, file_name)
                
                with open(file_location, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                # Handle OCR detection for PDFs
                use_ocr = False
                ocr_status = None
                
                if file_ext == '.pdf':
                    if is_scanned:
                        ocr_check = get_ocr_status()
                        if ocr_check.get("tesseract_installed", False):
                            use_ocr = True
                            ocr_status = "Tesseract OCR will be applied"
                        else:
                            ocr_status = "WARNING: Tesseract OCR not installed"
                    else:
                        # Auto-detect
                        try:
                            is_scanned_pdf, detection_info = detect_scanned_pdf(file_location)
                            if is_scanned_pdf:
                                ocr_check = get_ocr_status()
                                if ocr_check.get("tesseract_installed", False):
                                    use_ocr = True
                                    ocr_status = "Auto-detected scanned PDF - Tesseract OCR will be applied"
                        except Exception as e:
                            logger.warning(f"Could not auto-detect PDF type for {file.filename}: {str(e)}")
                
                # Parse document
                logger.info(f"Parsing file: {file.filename} (session: {session_id})")
                documents = parse_document(file_location, use_ocr=use_ocr)
                
                if not documents:
                    failed_files.append({
                        "file_name": file.filename,
                        "error": "No content could be extracted"
                    })
                    continue
                
                # Ensure metadata includes session_id, source_file, and citation with original filename
                for doc in documents:
                    doc.metadata["session_id"] = session_id
                    doc.metadata["source_file"] = file.filename  # Original filename
                    # Update citation to use original filename (not timestamped stored name)
                    page_num = doc.metadata.get("page_number")
                    if page_num:
                        citation = f"{file.filename}, Page {page_num}"
                    else:
                        citation = file.filename
                    doc.metadata["citation"] = citation
                
                # Add to SESSION FAISS (incremental)
                success = session_manager.add_uploaded_file_embeddings(
                    documents, 
                    source_file=file.filename
                )
                
                if success:
                    processed_files.append({
                        "file_name": file.filename,
                        "stored_name": file_name,
                        "file_type": file_ext,
                        "chunks": len(documents),
                        "ocr_status": ocr_status
                    })
                    total_chunks += len(documents)
                    if ocr_status:
                        ocr_statuses[file.filename] = ocr_status
                else:
                    failed_files.append({
                        "file_name": file.filename,
                        "error": "Failed to create embeddings"
                    })
                    
            except Exception as e:
                logger.exception(f"Error processing file {file.filename}: {str(e)}")
                failed_files.append({
                    "file_name": file.filename,
                    "error": str(e)
                })
        
        # Update session metadata
        from services.session_storage import save_session_metadata
        save_session_metadata(session_id, session_manager.metadata)
        
        response_data = {
            "status": "success" if processed_files else "partial" if failed_files else "failed",
            "message": f"Processed {len(processed_files)} file(s), {len(failed_files)} failed",
            "session_id": session_id,
            "total_files": len(files),
            "processed": len(processed_files),
            "failed": len(failed_files),
            "total_chunks": total_chunks,
            "files": processed_files,
            "failed_files": failed_files if failed_files else None,
            "ocr_statuses": ocr_statuses if ocr_statuses else None,
            "storage_location": "session"
        }
        
        logger.info(f"✓ Multi-file upload complete: {len(processed_files)}/{len(files)} files, {total_chunks} total chunks in session {session_id}")
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.exception("Error during multi-file upload")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/user/store-file")
async def store_file(request: ExcelFileInput = Depends()):
    try:
        # Get or create session_id
        session_id = request.session_id
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.info(f"Created new session for file upload: {session_id}")
        else:
            logger.info(f"Using existing session for file upload: {session_id}")
        
        UPLOAD_DIR = f"uploaded_excels/{request.pb_number}"
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # Get supported formats dynamically
        supported_formats = get_supported_formats()
        file_ext = os.path.splitext(request.file.filename)[1].lower()
        
        if file_ext not in supported_formats:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Unsupported file format: {file_ext}",
                    "supported_formats": supported_formats
                }
            )

        file_name = f"{os.path.splitext(request.file.filename)[0]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}{os.path.splitext(request.file.filename)[1]}"
        file_location = os.path.join(UPLOAD_DIR, file_name)

        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(request.file.file, buffer)

        # Handle OCR if PDF and is_scanned flag is set
        ocr_status = None
        use_ocr = False
        
        # Check if file is PDF and needs OCR
        if file_ext == '.pdf':
            if request.is_scanned:
                # User explicitly marked as scanned
                from services.ocr_service import get_ocr_status, is_tesseract_installed
                ocr_check = get_ocr_status()
                logger.info(f"📄 PDF file with is_scanned=True: {file_name}")
                logger.info(f"Tesseract status: available={ocr_check.get('tesseract_available', False)}, installed={ocr_check.get('tesseract_installed', False)}")
                
                if ocr_check.get("tesseract_installed", False):
                    logger.info(f"📄 Scanned PDF detected: {file_name} - Tesseract OCR will be applied during parsing")
                    ocr_status = f"Tesseract OCR will be applied during indexing"
                    use_ocr = True
                else:
                    logger.warning("Tesseract OCR not installed. Cannot process scanned PDF.")
                    ocr_status = "WARNING: Tesseract OCR not installed - install: apt-get install tesseract-ocr (Linux) or brew install tesseract (Mac)"
            else:
                # Auto-detect if PDF is scanned (even if is_scanned=False)
                from services.ocr_service import detect_scanned_pdf, get_ocr_status, is_tesseract_installed
                try:
                    is_scanned, detection_info = detect_scanned_pdf(file_location)
                    if is_scanned:
                        logger.info(f"📄 Auto-detected scanned PDF: {file_name} (confidence: {detection_info.get('confidence', 'unknown')})")
                        ocr_check = get_ocr_status()
                        if ocr_check.get("tesseract_installed", False):
                            logger.info("Auto-enabling Tesseract OCR for scanned PDF")
                            ocr_status = f"Auto-detected scanned PDF - Tesseract OCR will be applied"
                            use_ocr = True
                        else:
                            logger.warning("Scanned PDF detected but Tesseract OCR not available")
                            ocr_status = "WARNING: Scanned PDF detected but Tesseract OCR not available"
                except Exception as e:
                    logger.warning(f"Could not auto-detect PDF type: {str(e)}")

        # Parse document and create embeddings for SESSION_FAISS
        from services.document_parser import parse_document
        from services.session_faiss_manager import SessionFAISSManager
        from services.multimodal_embeddings import get_multimodal_embeddings
        
        logger.info(f"Parsing document and creating embeddings for session: {session_id}")
        logger.info(f"  File: {file_name}")
        logger.info(f"  Use OCR: {use_ocr}")
        logger.info(f"  File extension: {file_ext}")
        logger.info(f"  File path: {file_location}")
        
        try:
            documents = parse_document(file_location, use_ocr=use_ocr)
            logger.info(f"✓ Parsed {len(documents)} document chunks from {file_name}")
            
            # Log content preview for debugging
            if documents:
                total_chars = sum(len(doc.page_content) for doc in documents)
                logger.info(f"  Total characters extracted: {total_chars}")
                logger.info(f"  First chunk preview: {documents[0].page_content[:100]}...")
                
                # Check if we got very little content
                if total_chars < 50:
                    logger.warning(f"⚠️  WARNING: Only {total_chars} characters extracted from {file_name}")
                    logger.warning(f"   This might be a scanned PDF. Try uploading with is_scanned=True")
            else:
                logger.error(f"✗ No documents extracted from {file_name}")
            
            if not documents:
                logger.warning(f"No documents extracted from {file_name}")
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "No content could be extracted from the file",
                        "session_id": session_id,
                        "file_name": file_name,
                        "use_ocr": use_ocr
                    }
                )
            
            # Log document details
            for idx, doc in enumerate(documents[:3]):  # Log first 3 documents
                logger.debug(f"  Document {idx}: {len(doc.page_content)} chars, metadata: {list(doc.metadata.keys())}")
            
        except Exception as e:
            logger.exception(f"Error parsing document {file_name}: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": f"Error parsing document: {str(e)}",
                    "session_id": session_id,
                    "file_name": file_name
                }
            )
        
        # Initialize SessionFAISSManager
        logger.info("Initializing embeddings model for session FAISS...")
        try:
            embeddings = get_multimodal_embeddings(model_path="./all-MiniLM-L6-v2", device='cpu')
            logger.info("✓ Embeddings model loaded")
        except Exception as e:
            logger.exception(f"Error loading embeddings model: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": f"Error loading embeddings model: {str(e)}",
                    "session_id": session_id
                }
            )
        
        try:
            session_manager = SessionFAISSManager(session_id, embeddings)
            logger.info(f"✓ SessionFAISSManager initialized for session: {session_id}")
        except Exception as e:
            logger.exception(f"Error initializing SessionFAISSManager: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": f"Error initializing session manager: {str(e)}",
                    "session_id": session_id
                }
            )
        
            # Ensure metadata includes session_id, source_file, and citation with original filename
            original_filename = request.file.filename
            for doc in documents:
                doc.metadata["session_id"] = session_id
                doc.metadata["source_file"] = original_filename  # Original filename (not timestamped)
                # Update citation to use original filename
                page_num = doc.metadata.get("page_number")
                if page_num:
                    citation = f"{original_filename}, Page {page_num}"
                else:
                    citation = original_filename
                doc.metadata["citation"] = citation
            
            # Add uploaded file embeddings to SESSION_FAISS (incremental)
            logger.info(f"Creating embeddings for {len(documents)} document chunks...")
            try:
                success = session_manager.add_uploaded_file_embeddings(
                    documents, 
                    source_file=original_filename  # Use original filename, not timestamped
                )
                
                if not success:
                    logger.error(f"Failed to create embeddings - add_uploaded_file_embeddings returned False")
                    return JSONResponse(
                        status_code=500,
                        content={
                            "error": "Failed to create embeddings for the uploaded file",
                            "session_id": session_id,
                            "document_count": len(documents)
                        }
                    )
                
                logger.info(f"✓ Successfully created embeddings for {len(documents)} chunks")
                
            except Exception as e:
                logger.exception(f"Exception during embedding creation: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": f"Exception during embedding creation: {str(e)}",
                        "session_id": session_id,
                        "document_count": len(documents)
                    }
                )
        
        # Update session metadata
        session_manager.metadata.uploaded_file_name = request.file.filename
        session_manager.metadata.uploaded_file_type = file_ext
        from services.session_storage import save_session_metadata
        save_session_metadata(session_id, session_manager.metadata)

        response_data = {
            "status": "success",
            "message": "File uploaded and embedded in session FAISS",
            "session_id": session_id,
            "file_name": file_name,
            "file_location": file_location,
            "file_type": file_ext,
            "chunks_stored": len(documents),
            "storage_location": "session",
            "is_scanned": request.is_scanned,
            "ocr_status": ocr_status
        }
        
        logger.info(f"✓ File uploaded and embedded: {file_name} ({len(documents)} chunks) in session {session_id}")
        if ocr_status:
            logger.info(f"OCR Status: {ocr_status}")
            
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.exception("Error during file upload and embedding")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    
@app.post("/user/files")
async def send_file_names(request: NamesReq):
    try:
        folder_path = os.path.join("uploaded_excels", request.pb_number)

        if not os.path.exists(folder_path):
            return JSONResponse(status_code=404, content={"error": "Directory not found."})

        # List all supported file formats
        supported_formats = get_supported_formats()
        
        all_files = [
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in supported_formats and os.path.isfile(os.path.join(folder_path, f))
        ]

        return {
            "files": all_files,
            "supported_formats": supported_formats
        }
    
    except Exception as e:
        logger.exception("Error retrieving file names")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/user/analytics")
async def analyse(request: QueryRequestFile) -> Dict[Any, Any]:
    try:
        filename = request.file_name
        pb_number = request.pb_number
        final_query = request.query
        
        # Enhanced prompt verification
        is_valid, error_msg, verification_details = verify_prompt(final_query, context="aircraft")
        if not is_valid:
            return {
                "error": error_msg,
                "verification_details": verification_details
            }

        if filename == "default":
            print("🚁 Aircraft Snag Resolution System - JSON Output")
            print("=" * 50)

            chain, db = get_analytics_chain()

            final_query = request.query 

            print("🔍 Final LLM Query:\n", final_query)
        else:
            chain, db = get_analytics_chain_from_xls(filename,pb_number)
        # Get AI-generated rectification
        print("🔍 Final LLM Query:\n", final_query)

        json_results = process_snag_query_json_analysis(chain, db, final_query)
        return convert_numpy(json_results)

    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# NEW ENDPOINTS FOR ENHANCED FEATURES
# ============================================================================

@app.get("/system/ocr-status")
async def get_ocr_status_endpoint():
    """Get OCR system status and check if Tesseract is installed."""
    from services.ocr_service import get_ocr_status
    
    status = get_ocr_status()
    return {
        "ocr_available": status["tesseract_installed"],
        "ocr_engine": "Tesseract",
        "tesseract_version": status.get("tesseract_version", None),
        "supported_languages": status.get("supported_languages", []),
        "details": status,
        "instructions": {
            "linux": "sudo apt-get install tesseract-ocr",
            "mac": "brew install tesseract",
            "windows": "Download from: https://github.com/UB-Mannheim/tesseract/wiki"
        }
    }


@app.post("/system/detect-scanned-pdf")
async def detect_scanned_pdf_endpoint(file: UploadFile = File(...)):
    """
    Detect if an uploaded PDF is scanned (image-only) or has extractable text.
    Helps users determine if they need to enable OCR mode.
    """
    from services.ocr_service import detect_scanned_pdf
    import tempfile
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Detect PDF type
        is_scanned, detection_info = detect_scanned_pdf(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        return {
            "file_name": file.filename,
            "is_scanned": is_scanned,
            "recommendation": "Enable OCR mode when uploading" if is_scanned else "Use normal mode (no OCR needed)",
            "detection_details": detection_info
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/system/info")
async def system_info():
    """Get system information including supported formats and capabilities."""
    format_support = check_format_support()
    supported_formats = get_supported_formats()
    
    return {
        "service": "Sky-Sentinel RAG Knowledge Portal",
        "version": "2.0.0",
        "features": {
            "multi_format_support": True,
            "offline_operation": True,
            "citations": True,
            "traceability": True,
            "incremental_learning": True,
            "rbac": True,
            "prompt_verification": True
        },
        "supported_formats": {
            "pdf": format_support["pdf"],
            "docx": format_support["docx"],
            "txt": format_support["txt"],
            "excel": format_support["excel"]
        },
        "file_extensions": supported_formats,
        "llm_model": "llama3.2:8b-instruct-q4_K_M (recommended)",
        "embedding_model": "all-MiniLM-L6-v2"
    }


@app.post("/user/verify-query")
async def verify_query_endpoint(request: QueryRequest):
    """Verify query quality and relevance before processing."""
    is_valid, message, details = verify_prompt(request.query, context="aircraft")
    
    return {
        "is_valid": is_valid,
        "message": message,
        "verification_details": details,
        "recommendations": _get_query_recommendations(details) if not is_valid else None
    }


@app.get("/admin/index/statistics")
async def get_index_statistics():
    """Get statistics about the vector store index."""
    try:
        manager = IncrementalLearningManager("snag_faiss_index")
        stats = manager.get_statistics()
        sources = manager.list_sources()
        
        return {
            "statistics": stats,
            "sources": sources[:20],  # Limit to 20 for response size
            "total_sources": len(sources)
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/index/add_document")
async def add_document_to_index(
    file: UploadFile = File(...),
    pb_number: str = Form(...),
    department: Optional[str] = Form(None)
):
    """Add a new document to the vector store incrementally."""
    try:
        # Save uploaded file
        UPLOAD_DIR = f"uploaded_excels/{pb_number}"
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        file_name = f"{os.path.splitext(file.filename)[0]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}{os.path.splitext(file.filename)[1]}"
        file_location = os.path.join(UPLOAD_DIR, file_name)
        
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Use ingestion service with department routing
        result = ingest_single_file(
            file_path=file_location,
            department=department,
            incremental=True
        )
        
        if result.get("success"):
            return {
                "message": "Document added successfully",
                "file_name": file_name,
                "num_chunks": result.get("num_chunks", 0),
                "file_type": os.path.splitext(file_name)[1],
                "department": department,
                "index_path": result.get("index_path"),
                "statistics": get_index_info(result.get("index_path", "snag_faiss_index"))
            }
        else:
            return JSONResponse(
                status_code=500,
                content={"error": result.get("error", "Failed to add document to index")}
            )
            
    except Exception as e:
        logger.exception("Error adding document to index")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/system/formats")
async def get_supported_formats_endpoint():
    """Get list of supported document formats."""
    format_support = check_format_support()
    supported_formats = get_supported_formats()
    
    return {
        "supported_formats": supported_formats,
        "format_details": {
            "pdf": {
                "supported": format_support["pdf"],
                "description": "PDF documents with page-level citations",
                "install_command": "pip install pymupdf" if not format_support["pdf"] else None
            },
            "docx": {
                "supported": format_support["docx"],
                "description": "Microsoft Word documents with paragraph-level citations",
                "install_command": "pip install python-docx" if not format_support["docx"] else None
            },
            "txt": {
                "supported": format_support["txt"],
                "description": "Plain text files with line-level citations",
                "install_command": None
            },
            "excel": {
                "supported": format_support["excel"],
                "description": "Excel spreadsheets with row-level citations",
                "install_command": None
            }
        }
    }


def _get_query_recommendations(verification_details: Dict[str, Any]) -> List[str]:
    """Get recommendations for improving query quality."""
    recommendations = []
    
    if not verification_details.get("length_check"):
        recommendations.append("Provide more context in your query (at least 10 characters)")
    
    if not verification_details.get("semantic_check"):
        recommendations.append("Use more descriptive words about the aircraft issue or topic")
    
    if not verification_details.get("relevance_check"):
        recommendations.append("Include aircraft-related terms (e.g., engine, hydraulic, system, maintenance)")
    
    quality_score = verification_details.get("quality_score", 0)
    if quality_score < 0.5:
        recommendations.append("Try to be more specific about the problem or question")
    
    return recommendations


# ============================================================================
# ADMIN ENDPOINTS FOR GLOBAL INDEX MANAGEMENT
# ============================================================================

@app.post("/admin/ingest/file")
async def admin_ingest_file(
    file: UploadFile = File(...),
    incremental: bool = Form(True),
    index_path: str = Form("snag_faiss_index"),
    department: Optional[str] = Form(None)
):
    """
    Admin endpoint: Ingest a single file into the global FAISS index.
    Supports PDF, DOCX, TXT, XLS, XLSX.
    If department is provided, stores in department-specific path: snag_faiss_index/{department}/faiss_index
    """
    try:
        # Save uploaded file temporarily
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        file_name = file.filename
        temp_path = os.path.join(temp_dir, file_name)
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Admin ingesting file: {file_name} (department: {department or 'none'})")
        
        # Ingest file with department routing
        result = ingest_single_file(
            file_path=temp_path,
            index_path=index_path,
            incremental=incremental,
            department=department
        )
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
        
        if result["success"]:
            return {
                "success": True,
                "message": f"Successfully ingested {file_name}",
                "details": result
            }
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": result.get("error", "Unknown error")
                }
            )
            
    except Exception as e:
        logger.exception("Error in admin file ingestion")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


@app.post("/admin/ingest/directory")
async def admin_ingest_directory(
    directory_path: str = Form(...),
    recursive: bool = Form(True),
    index_path: str = Form("snag_faiss_index"),
    department: Optional[str] = Form(None)
):
    """
    Admin endpoint: Ingest all supported files from a directory.
    Useful for batch ingestion of multiple documents.
    If department is provided, stores in department-specific path.
    """
    try:
        logger.info(f"Admin ingesting directory: {directory_path} (department: {department or 'none'})")
        
        result = ingest_directory(
            directory_path=directory_path,
            index_path=index_path,
            recursive=recursive,
            department=department
        )
        
        if result["success"]:
            return {
                "success": True,
                "message": f"Processed {result['processed']} files successfully",
                "details": result
            }
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": result.get("error", "Unknown error")
                }
            )
            
    except Exception as e:
        logger.exception("Error in admin directory ingestion")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


@app.post("/admin/index/rebuild")
async def admin_rebuild_index(
    source_paths: List[str] = Form(...),
    index_path: str = Form("snag_faiss_index"),
    department: Optional[str] = Form(None)
):
    """
    Admin endpoint: Rebuild the entire FAISS index from scratch.
    WARNING: This will delete the existing index and rebuild it.
    If department is provided, rebuilds department-specific index.
    """
    try:
        logger.info(f"Admin rebuilding index from {len(source_paths)} sources (department: {department or 'none'})")
        
        result = rebuild_index_from_scratch(
            source_paths=source_paths,
            index_path=index_path,
            department=department
        )
        
        if result["success"]:
            return {
                "success": True,
                "message": f"Index rebuilt with {result['total_documents']} documents",
                "details": result
            }
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": result.get("error", "Unknown error")
                }
            )
            
    except Exception as e:
        logger.exception("Error in admin index rebuild")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


@app.get("/pdfs/{filename}")
async def serve_pdf(filename: str):
    """
    Serve PDF files for citation viewing.
    Searches in uploaded_excels directories and data directory.
    """
    # Search in common locations
    search_paths = [
        f"data/pdfs/{filename}",
        f"uploaded_excels/**/{filename}",
    ]
    
    # Also search in all pb_number directories
    for pb_dir in os.listdir("uploaded_excels"):
        pdf_path = f"uploaded_excels/{pb_dir}/{filename}"
        if os.path.exists(pdf_path):
            return FileResponse(
                pdf_path,
                media_type="application/pdf",
                filename=filename
            )
    
    # Check data/pdfs
    pdf_path = f"data/pdfs/{filename}"
    if os.path.exists(pdf_path):
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=filename
        )
    
    raise HTTPException(status_code=404, detail=f"PDF file not found: {filename}")


@app.get("/api/citation/{citation_id}")
async def get_citation_details(citation_id: str, session_id: str = None):
    """
    Get detailed citation information by citation ID.
    Used when user clicks a citation in the UI.
    
    Args:
        citation_id: Citation ID (e.g., "cite_1")
        session_id: Optional session ID to filter citations
    
    Returns:
        Citation details with PDF coordinates
    """
    try:
        # If session_id provided, get citations from that session
        if session_id:
            citations = get_citations_for_session(session_id)
            citation = get_citation_by_id(citation_id, citations)
            
            if citation:
                # Add file_url for serving
                filename = citation["source"]["file"]
                citation["source"]["file_url"] = f"/pdfs/{filename}"
                return citation
        
        # Fallback: return error
        return JSONResponse(
            status_code=404,
            content={
                "error": f"Citation {citation_id} not found",
                "session_id": session_id
            }
        )
        
    except Exception as e:
        logger.exception("Error retrieving citation details")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/api/test-pdf-bbox")
async def test_pdf_bbox_extraction(
    file: UploadFile = File(...),
    page_num: int = Form(...),
    search_text: str = Form(...)
):
    """
    Test endpoint to verify PDF bbox extraction works.
    Upload a PDF and test text location finding.
    """
    from services.pdf_citation_service import test_bbox_extraction
    import tempfile
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Test bbox extraction
        results = test_bbox_extraction(temp_path, page_num, search_text)
        
        # Clean up
        os.remove(temp_path)
        
        return {
            "file_name": file.filename,
            "page_number": page_num,
            "search_text": search_text,
            "results": results,
            "bbox_found": results["bbox"] is not None
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/admin/index/info")
async def admin_get_index_info(index_path: str = "snag_faiss_index"):
    """
    Admin endpoint: Get information about the global FAISS index.
    Returns statistics, sources, and metadata.
    """
    try:
        result = get_index_info(index_path)
        
        if result.get("exists"):
            return {
                "success": True,
                "index_info": result
            }
        else:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": result.get("error", "Index not found")
                }
            )
            
    except Exception as e:
        logger.exception("Error getting index info")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )
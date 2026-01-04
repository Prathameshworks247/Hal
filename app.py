# app.py
from datetime import datetime
import os
import logging
import cv2
import numpy as np
import os
import io
import uuid
import shutil
from collections import defaultdict
import pandas as pd
import re
from fastapi.responses import StreamingResponse,JSONResponse
from fastapi import FastAPI, File, UploadFile, HTTPException, Form,Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Any
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
from services.incremental_learning import IncrementalLearningManager, create_or_update_index
from services.rbac_service import get_rbac_manager, Role, Permission
from services.similarity_service import  get_similar_records_with_metadata
from services.parsers import process_snag_query_json, display_results_as_json
from utils.utils import test_retriever, convert_numpy
from services.chain_service import get_analytics_chain
from services.parsers import process_snag_query_json_analysis
from services.chain_service import get_analytics_chain_from_xls
from services.similarity_service import has_semantic_meaning
from vision.funcs import detect_shapes,get_pixels_per_mm,save_to_csv

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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Shape Detection API"}

@app.on_event("startup")
async def cleanup_old_files():
    """Clean up old files on server startup"""
    try:
        for file in os.listdir("static"):
            if file.startswith(("processed_", "shapes_")):
                os.remove(f"static/{file}")
    except:
        pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@lru_cache()
def get_chain_cached():
    return get_chain()
def get_chain_file_chached(file_name, pb_number):
    return get_chain_file(file_name, pb_number)


@app.post("/rectify")
async def rectification(request: QueryRequestFile) -> Dict[Any, Any]:
    try:
        file_name = request.file_name
        pb_number = request.pb_number
        final_query = request.query
        
        # Enhanced prompt verification
        is_valid, error_msg, verification_details = verify_prompt(final_query, context="aircraft")
        if not is_valid:
            logger.warning(f"Query verification failed: {error_msg}")
            return {
                "error": error_msg,
                "verification_details": verification_details
            }
        
        logger.info(f"Query verified successfully. Quality score: {verification_details.get('quality_score', 0):.2f}")
        
        # Legacy snag extraction for backward compatibility
        match = re.search(r"Snag:\s*(.*?)(\s+\w+:|$)", final_query)
        if match:
            snag_text = match.group(1).strip()
            # Additional semantic check on snag text
            if not has_semantic_meaning(snag_text):
                print(f"=========={False}: {snag_text}============")
                return {"error": "Snag description lacks semantic meaning. Please provide more details."}

        print("🚁 Aircraft Snag Resolution System - JSON Output")
        if file_name == 'default':    
            chain, db = get_chain_cached()
            if os.getenv("DEBUG_MODE") == "1":
                test_retriever(db, "hydraulic system pressure low")

            print("🔍 Final LLM Query:\n", final_query)

            json_results = process_snag_query_json(chain, db, final_query)

            return jsonable_encoder(convert_numpy(json_results))

        else:
            chain, db = get_chain_file_chached(file_name, pb_number)
            response = chain.invoke({"question": final_query})

        # Extract result
            if isinstance(response, dict):
                rectification = response.get('result', response.get('answer', str(response)))
            else:
                rectification = str(response)
            
            similar_snags = get_similar_records_with_metadata(db, final_query, k=5)
            json_results = display_results_as_json(rectification, similar_snags, final_query)
            
            return jsonable_encoder(convert_numpy(json_results))

    except Exception as e:
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


    
@app.post("/get_unique_row", response_model=Dict[str, List[str]])
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

@app.post("/store_file")
async def store_file(request: ExcelFileInput = Depends()):
    try:
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

        print("File Uploaded:", file_location)
        return "File Uploaded Successfully"
    except Exception as e:
        logger.exception("Error during sending file")
        print("Error during sending file:", e)
        return {"error": str(e)}
    
@app.post("/send_file_names")
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


@app.post("/analytics")
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


@app.post("/verify_query")
async def verify_query_endpoint(request: QueryRequest):
    """Verify query quality and relevance before processing."""
    is_valid, message, details = verify_prompt(request.query, context="aircraft")
    
    return {
        "is_valid": is_valid,
        "message": message,
        "verification_details": details,
        "recommendations": _get_query_recommendations(details) if not is_valid else None
    }


@app.get("/index/statistics")
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
    pb_number: str = Form(...)
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
        
        # Parse document
        documents = parse_document(file_location)
        
        # Add to vector store incrementally
        manager = IncrementalLearningManager("snag_faiss_index")
        success = manager.add_documents(documents, source_file=file_location)
        
        if success:
            return {
                "message": "Document added successfully",
                "file_name": file_name,
                "num_chunks": len(documents),
                "file_type": os.path.splitext(file_name)[1],
                "statistics": manager.get_statistics()
            }
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to add document to index"}
            )
            
    except Exception as e:
        logger.exception("Error adding document to index")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/formats/supported")
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


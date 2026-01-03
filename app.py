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
from models.models import ExcelFileInput,GetRows,NamesReq,QueryRequestFile,ShapeDetectionResponse
from services.chain_service import get_chain, get_chain_file,verify
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
        # verdict = verify(final_query)
        # print("The Query Is: ", verdict)
        # if not verdict:
        #     return  {"error": "please enter a valid query"}
        match = re.search(r"Snag:\s*(.*?)(\s+\w+:|$)", final_query)
        if not match:
            return {"error": "Could not find Snag in the query"}

        snag_text = match.group(1).strip()  # extract only the snag part
        verdict = has_semantic_meaning(snag_text)
        if not verdict:
            print(f"=========={verdict}: {snag_text}============")
            return {"error": "please enter a valid query"}

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

        if not request.file.filename.endswith(('.xlsx', '.xls')):
            return JSONResponse(
                status_code=400,
                content={"error": "Only .xlsx or .xls files are allowed."}
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

        # List only Excel files
        excel_files = [
            f for f in os.listdir(folder_path)
            if f.endswith(('.xlsx', '.xls')) and os.path.isfile(os.path.join(folder_path, f))
        ]

        return {"files": excel_files}
    
    except Exception as e:
        logger.exception("Error retrieving file names")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/analytics")
async def analyse(request: QueryRequestFile) -> Dict[Any, Any]:
    try:
        filename = request.file_name
        pb_number = request.pb_number
        final_query = request.query

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


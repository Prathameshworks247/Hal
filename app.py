# app.py
from datetime import datetime
import os
import logging
from collections import defaultdict
from fastapi.responses import JSONResponse
import pandas as pd
from fastapi import FastAPI,Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document
from collections import defaultdict
from services.llm import get_llm
from fastapi.encoders import jsonable_encoder
from functools import lru_cache
import shutil
from models.models import QueryRequest, ExcelFileInput,GetRows,NamesReq,QueryRequestFile
from services.chain_service import get_chain, get_chain_file,verify
from services.similarity_service import  get_similar_records_with_metadata
from services.excel_service import excel_to_documents
from services.parsers import process_snag_query_json, display_results_as_json
from utils.utils import test_retriever, convert_numpy

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        verdict = verify(final_query)
        print("The Query Is: ", verdict)
        if not verdict:
            return  {"error": "please enter a valid query"}
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
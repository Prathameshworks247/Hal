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
from services.chain_service import get_chain
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


@app.post("/rectify")
async def rectification(request: QueryRequest) -> Dict[Any, Any]:
    try:
        print("🚁 Aircraft Snag Resolution System - JSON Output")
        print("=" * 50)

        chain, db = get_chain_cached()

        if os.getenv("DEBUG_MODE") == "1":
            test_retriever(db, "hydraulic system pressure low")

        final_query = request.query 

        print("🔍 Final LLM Query:\n", final_query)

        json_results = process_snag_query_json(chain, db, final_query)

        return jsonable_encoder(convert_numpy(json_results))

    except Exception as e:
        return {"error": str(e)}

@app.post("/rectify-file")
async def rectify_file(request: QueryRequestFile) -> Dict[Any, Any]:
    try:
        parts = [
            f"Query: {request.query}"
        ]
        UPLOAD_DIR = f"uploaded_excels/{request.pb_number}"
        final_query = request.query
        file_location = os.path.join(UPLOAD_DIR, request.file_name)
        docs = excel_to_documents(file_location)
        if not docs:
            return {"error": "No relevant historical snag records found."}
        model_path = "./all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": "cpu"}
        )

        prompt = PromptTemplate.from_template("""
        You are an expert aircraft technician with extensive experience in aircraft maintenance and troubleshooting.
        
        Based on the following historical snag records and their rectifications, provide a detailed recommendation for fixing the current snag.
        
        Current Snag: {question}
        
        Historical Snag Records:
        {context}
        
        Please provide:
        1. Most likely cause of the issue
        2. Step-by-step rectification procedure
        3. Any safety precautions to consider
        4. Parts that might need replacement
        5. Expected time to complete the fix
        
        Recommended Rectification:
        """)

        vectorstore = FAISS.from_documents(
            docs,
            embedding=embeddings,
        )

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        logger.info("Getting LLM instance...")
        llm = get_llm()
        logger.info("LLM instance obtained successfully.")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt, "verbose": True},
            return_source_documents=True,
            input_key="question",
            output_key="result"
        )
        logger.info(f"Processing query: {final_query}")
        
        # Get AI-generated rectification
        response = qa_chain.invoke({"question": final_query})

        # Extract result
        if isinstance(response, dict):
            rectification = response.get('result', response.get('answer', str(response)))
        else:
            rectification = str(response)
        
        similar_snags = get_similar_records_with_metadata(vectorstore, final_query, k=5)
        json_results = display_results_as_json(rectification, similar_snags, final_query)
        
        return jsonable_encoder(convert_numpy(json_results))


    except Exception as e:
        logger.exception("Error during rectification")
        return {"error": str(e)}


    
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
            if key.strip().lower() == "rectification":
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
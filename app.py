# app.py
from datetime import datetime
import os
import logging
from collections import defaultdict
from fastapi.responses import JSONResponse
import pandas as pd
from fastapi import FastAPI, Request, UploadFile, File,Form,Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Tuple, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document
from collections import defaultdict
from langchain.schema import Document 
from llm import get_llm
import numpy as np
import json
from fastapi.encoders import jsonable_encoder

from functools import lru_cache
import shutil
import re

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

def get_chain():
    try:
        logger.info("Initializing embeddings model...")
        model_path = "./all-MiniLM-L6-v2"
        if not os.path.exists(model_path):
            logger.warning(f"Local model path {model_path} not found.")
            return
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cpu'} 
        )
        logger.info("Embeddings model loaded successfully.")

        logger.info("Loading FAISS index...")
        # Check if FAISS index exists
        faiss_index_path = "snag_faiss_index"
        if not os.path.exists(faiss_index_path):
            raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")
        
        db = FAISS.load_local(
            faiss_index_path, 
            embeddings=embeddings, 
            allow_dangerous_deserialization=True
        )
        logger.info("FAISS index loaded successfully.")

        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5}) 
        logger.info("Retriever configured successfully.")

        
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
        
        ---
        Rectification:
        [Provide your detailed rectification steps here]
    
        """)
        logger.info("Getting LLM instance...")
        llm = get_llm()
        logger.info("LLM instance obtained successfully.")

        # Create the QA chain with proper input/output keys
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": prompt,
                "verbose": True  # Enable verbose mode for debugging
            },
            return_source_documents=True,  # Return source documents for transparency
            input_key="question",  # Explicitly set input key
            output_key="result"    # Explicitly set output key
        )
        
        logger.info("QA chain created successfully.")
        return qa_chain, db

    except Exception as e:
        logger.error(f"Error in get_chain(): {str(e)}")
        raise


@lru_cache()
def get_chain_cached():
    return get_chain()


logger = logging.getLogger(__name__)

def get_similar_snags_with_metadata(db, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve similar snags with their metadata and similarity scores
    
    Args:
        db: FAISS database instance
        query: Current snag description
        k: Number of similar documents to retrieve
        
    Returns:
        List of dictionaries containing document content, metadata, and similarity scores
    """
    try:
        # Get similar documents with scores
        docs_with_scores = db.similarity_search_with_score(query, k=k)

        similar_snags = []
        for i, (doc, score) in enumerate(docs_with_scores):

            snag_match = re.search(r"SNAG:\s*(.*)", doc.page_content)
            rect_match = re.search(r"RECTIFICATION:\s*(.*)", doc.page_content)

            snag_info = {
                'rank': i + 1,
                'snag': snag_match.group(1).strip() if snag_match else "",
                'rectification': rect_match.group(1).strip() if rect_match else "",
                'metadata': doc.metadata,
                'similarity_score': float(score),
                'similarity_percentage': round((1 - score) * 100, 2)
            }

            similar_snags.append(snag_info)

        return similar_snags

    except Exception as e:
        logger.error(f"Error retrieving similar snags: {str(e)}")
        return []

def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    return obj

def process_snag_query_json(chain, db, query: str) -> Dict[str, Any]:
    """
    Process snag query and return results in JSON format
    
    Args:
        chain: QA chain instance
        db: FAISS database instance
        query: Snag description
        
    Returns:
        Complete JSON response with rectification and similar snags
    """
    try:
        logger.info(f"Processing query: {query}")
        
        # Get AI-generated rectification
        response = chain.invoke({"question": query})
        
        # Extract result
        if isinstance(response, dict):
            rectification = response.get('result', response.get('answer', str(response)))
        else:
            rectification = str(response)
        
        # Get similar snags with metadata
        similar_snags = get_similar_snags_with_metadata(db, query, k=5)

        # Format as JSON
        json_results = display_results_as_json(rectification, similar_snags, query)
        
        return json_results
        
    except Exception as e:
        logger.error(f"Error processing snag query: {str(e)}")
        from datetime import datetime
        return {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "status": "error",
            "error_message": str(e),
            "rectification": None,
            "similar_historical_snags": [],
            "summary": {
                "total_similar_cases_found": 0,
                "average_similarity_percentage": 0,
                "highest_similarity_percentage": 0,
                "lowest_similarity_percentage": 0,
                "recommendation_reliability": "none"
            }
        }


# def extract_rectification_and_analytics(response_text: str) -> Dict[str, Any]:
#     import re
#     import json

#     result = {
#         "rectification": "",
#         "analytics": []
#     }

#     # Extract rectification
#     rect_match = re.search(
#         r"Rectification:\s*(.*?)\s*---", 
#         response_text, 
#         re.DOTALL | re.IGNORECASE
#     )
#     if rect_match:
#         result["rectification"] = rect_match.group(1).strip()
#     else:
#         print("⚠️ No rectification section found.")

#     # Extract and fix analytics
#     analytics_match = re.search(
#         r"Analytics\s*:\s*(\[[\s\S]*?\])", 
#         response_text, 
#         re.DOTALL | re.IGNORECASE
#     )
#     if analytics_match:
#         analytics_str = analytics_match.group(1).strip()
#         analytics_str = analytics_str.replace("{{", "{").replace("}}", "}")  # ✅ FIX HERE
#         try:
#             result["analytics"] = json.loads(analytics_str)
#         except json.JSONDecodeError as e:
#             print(f"⚠️ JSON decoding failed: {e}")
#     else:
#         print("⚠️ No analytics section found.")

#     return result



def display_results_as_json(response_text: str, similar_snags: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """Format and display results as structured JSON"""
    num_snags = len(similar_snags)
    similarity_scores = [s['similarity_score'] for s in similar_snags]

    avg_similarity = sum(similarity_scores) / num_snags if num_snags else 0

    results = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "status": "success",
        "rectification": {
            "ai_recommendation": response_text,
            "based_on_historical_cases": num_snags
        },
        "similar_historical_snags": similar_snags,
        "analytics": {
            "total_similar_cases_found": num_snags,
            "average_similarity_percentage": round(avg_similarity * 100, 2),
            "highest_similarity_percentage": round(max(similarity_scores) * 100, 2) if num_snags else 0,
            "lowest_similarity_percentage": round(min(similarity_scores) * 100, 2) if num_snags else 0,
            "recommendation_reliability": (
                "high" if num_snags >= 3 and similarity_scores[0] * 100 > 75
                else "medium" if num_snags >= 2
                else "low"
            ),
        }
    }

    return results


def test_retriever(db, query):
    """Test function to check if retriever is working"""
    try:
        print(f"\n🧪 Testing retriever with query: '{query}'")
        similar_snags = get_similar_snags_with_metadata(db, query, k=3)
        
        if similar_snags:
            print(f"✅ Retriever test successful! Found {len(similar_snags)} similar documents.")
            test_results = {
                "test_query": query,
                "status": "success",
                "documents_found": len(similar_snags),
                "sample_results": similar_snags[:2] 
            }
            print("\n📄 Test Results (JSON):")
            print(json.dumps(test_results, indent=2, ensure_ascii=False))
            return True
        else:
            print("❌ No documents retrieved.")
            return False
            
    except Exception as e:
        print(f"❌ Retriever test failed: {e}")
        return False

def export_results_to_dict(rectification: str, similar_snags: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Export results in a structured dictionary format for API usage or further processing
    """
    return {
        'rectification': {
            'recommendation': rectification,
            'generated_at': str(logging.time.time())
        },
        'similar_snags': similar_snags,
        'summary': {
            'total_similar_snags': len(similar_snags),
            'avg_similarity': round(sum(s['similarity_percentage'] for s in similar_snags) / len(similar_snags), 2) if similar_snags else 0,
            'most_similar_score': similar_snags[0]['similarity_percentage'] if similar_snags else 0
        }
    }

def excel_to_documents(file_path: str) -> list[Document]:
    df = pd.read_excel(file_path)
    documents = []

    for idx, row in df.iterrows():
        content_lines = [f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])]
        content = "\n".join(content_lines)

        documents.append(
            Document(
                page_content=content,
                metadata={"row_index": idx, "source": "excel_row"}
            )
        )
    return documents

def excel_columns(filepath):
    df = pd.read_excel(filepath)
    cols = df.columns
    lis_cols = cols.to_list()
    return lis_cols
    

class FlightHours(BaseModel):
    lower: int
    upper: int

class QueryRequest(BaseModel):
    query: str
    helicopter_type: Optional[str] = None
    flight_hours: Optional[FlightHours] = None
    event_type: Optional[str] = None
    status: Optional[str] = None
    raised_by: Optional[str] = None


class QueryRequestFile(BaseModel):
    query: str
    file_name: str

class ExcelFileInput:
    def __init__(
        self,
        file: UploadFile = File(...),
        pv_number: str = Form(...)
    ):
        self.file = file
        self.pv_number = pv_number


@app.post("/rectify")
async def rectification(request: QueryRequest) -> Dict[Any, Any]:
    try:
        print("🚁 Aircraft Snag Resolution System - JSON Output")
        print("=" * 50)

        chain, db = get_chain_cached()

        if os.getenv("DEBUG_MODE") == "1":
            test_retriever(db, "hydraulic system pressure low")

        parts = [
            f"Query: {request.query}",
            f"Helicopter Type: {request.helicopter_type}" if request.helicopter_type else "",
            f"Flight Hours: {request.flight_hours.lower} to {request.flight_hours.upper}" if request.flight_hours else "",
            f"Event Type: {request.event_type}" if request.event_type else "",
            f"Status: {request.status}" if request.status else "",
            f"Raised By: {request.raised_by}" if request.raised_by else "",
        ]
        final_query = "\n".join([part for part in parts if part.strip()])  

        print("🔍 Final LLM Query:\n", final_query)

        json_results = process_snag_query_json(chain, db, final_query)

        return jsonable_encoder(convert_numpy(json_results))

    except Exception as e:
        return {"error": str(e)}

@app.post("/rectify-file")
async def rectification(request: QueryRequestFile) -> Dict[Any, Any]:
    try:
        parts = [
            f"Query: {request.query}"
        ]
        final_query = "\n".join([part for part in parts if part.strip()])  
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
        json_results = process_snag_query_json(qa_chain, vectorstore, final_query)

        return jsonable_encoder(convert_numpy(json_results))


    except Exception as e:
        logger.exception("Error during rectification")
        return {"error": str(e)}


    
@app.post("/get_unique_row", response_model=Dict[str, List[str]])
def get_unique_row(pv_number: str = Form(...), filename:str = Form(...)):
    try:
        DIR = f"uploaded_excels/{pv_number}"
        df = pd.read_excel(f"{DIR}/{filename}")
        columns = list(df.columns)
        temp = defaultdict(list)
        dic = defaultdict(list)

        for column in columns:
            unique_vals = df[column].dropna().unique().tolist()
            temp[column] = [str(v) if not isinstance(v, (str, int, float, bool)) else v for v in unique_vals]
        for key, value in temp.items():
            if len(value) < 100:
                dic[key] = value
        return JSONResponse(content=dic)

    except Exception as e:
        logger.exception("Error retrieving unique column values")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/store_file")
async def store_file(request: ExcelFileInput = Depends()):
    try:
        UPLOAD_DIR = f"uploaded_excels/{request.pv_number}"
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
        return excel_columns(file_location)
    except Exception as e:
        logger.exception("Error during sending file")
        return {"error": str(e)}
    
@app.post("/send_file_names")
async def send_file_names(pv_number: str = Form(...)):
    try:
        folder_path = os.path.join("uploaded_excels", pv_number)

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
# app.py
import os
import logging
from collections import defaultdict
from fastapi.responses import JSONResponse
import pandas as pd
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Tuple, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document
from collections import defaultdict
from langchain.schema import Document  # Or adjust based on your actual import
from llm import get_llm
import numpy as np
import json
import shutil

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR  = "uploaded_excels"
os.makedirs(UPLOAD_DIR, exist_ok=True)  
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
        
        Recommended Rectification:
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
        # Get retriever
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
        
        # Get similar documents
        similar_docs = retriever.get_relevant_documents(query)
        
        # Also get similarity scores using similarity_search_with_score
        docs_with_scores = db.similarity_search_with_score(query, k=k)
        
        # Combine documents with scores and metadata
        similar_snags = []
        for i, (doc, score) in enumerate(docs_with_scores):
            snag_info = {
                'rank': i + 1,
                'content': doc.page_content,
                'metadata': doc.metadata if hasattr(doc, 'metadata') else {},
                'similarity_score': float(score),
                'similarity_percentage': round((1 - score) * 100, 2)  # Convert distance to similarity percentage
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

def display_results_as_json(rectification: str, similar_snags: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """Format and display results as JSON"""
    import json
    from datetime import datetime
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "status": "success",
        "rectification": {
            "ai_recommendation": rectification,
            "based_on_historical_cases": len(similar_snags)
        },
        "similar_historical_snags": similar_snags,
        "summary": {
            "total_similar_cases_found": len(similar_snags),
            "average_similarity_percentage": (round(sum(s['similarity_score'] for s in similar_snags) / len(similar_snags), 2))*100 if similar_snags else 0,
            "highest_similarity_percentage": (similar_snags[0]['similarity_score'])*100 if similar_snags else 0,
            "lowest_similarity_percentage": (similar_snags[-1]['similarity_score'])*100 if similar_snags else 0,
            "recommendation_reliability": "high" if len(similar_snags) >= 3 and (similar_snags[0]['similarity_percentage'] > 75) else "medium" if len(similar_snags) >= 2 else "low"
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


@app.post("/rectify")
async def rectification(request: QueryRequest) -> Dict[Any, Any]:
    try:
        print("🚁 Aircraft Snag Resolution System - JSON Output")
        print("=" * 50)

        chain, db = get_chain()

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

        return convert_numpy(json_results)

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

        return convert_numpy(json_results)

    except Exception as e:
        logger.exception("Error during rectification")
        return {"error": str(e)}


    
@app.post("/unique-columns", response_model=Dict[str, List[str]])
def get_all_unique_column_values():
    df = pd.read_excel("All_snags_data.xlsx")
    
    df["HELI_TYPE"] = df["HELI_NO"].apply(lambda x: x[:2] if isinstance(x, str) and x[:2] in ["CG", "IA", "ZD", "IN", "J "] else None)

    columns = list(df.columns)
    temp = defaultdict(list)
    dic = defaultdict(list)

    for column in columns:
        unique_vals = df[column].dropna().unique().tolist()
        temp[column] = [str(v) if not isinstance(v, (str, int, float, bool)) else v for v in unique_vals]

    for key, value in temp.items():
        if len(value) < 10:
            dic[key] = value

    return JSONResponse(content=dic)

@app.post("/excel-upload")
async def excel_upload(file: UploadFile = File(...)):
    if not file.filename.endswith(('.xlsx','.xls')):
        return JSONResponse(status_code=400, content={"error": "Only .xlsx or .xls files are allowed."})
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location,"wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print("File Uploaded!")
    return excel_columns(file_location)
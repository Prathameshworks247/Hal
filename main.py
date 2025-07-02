from datetime import datetime
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
from langchain.schema import Document 
from llm import get_llm
import numpy as np
from functools import lru_cache
import json
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

        
        prompt = PromptTemplate.from_template('''
                                              You are an expert aircraft technician and data analyst with extensive experience in aircraft maintenance and troubleshooting.

Based on the following historical snag records and their rectifications, analyze the current snag and return only structured analytics as per the format.

---
Current Snag:
{question}

Historical Snag Records:
{context}
---

🔧 TASK:
Analyze the current snag using the historical records and output only the following charts and stats in **strict JSON-like format** with no extra explanation or commentary.

---
📊 REQUIRED FORMAT:
(Respond ONLY in this format. Do not add anything outside this block.)

---
Radar Chart : {{
    "Complexity": [1-5],
    "Time Needed": [1-5],
    "Tools Required": [1-5],
    "Risk Level": [1-5],
    "Frequency": [1-5]
}}
---
Similarity : [0-100]
---
Pie Chart : {{
    "category 1": [1-5],
    "category 2": [1-5],
    "category 3": [1-5]
    // Add more categories if needed
}}
---
Bar Chart 1 : {{
    "PLT": [integer],
    "GR": [integer],
    "Ground Observation": [integer]
}}
---
Bar Chart 2 : {{
    "IA": [integer],
    "J": [integer],
    "ZD": [integer],
    "IN": [integer],
    "CG": [integer]
}}
---

🚫 Do NOT include any explanations, comments, or extra text. Return only the formatted blocks as shown above.

                                              ''')
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
            analytics = response.get('result', response.get('answer', str(response)))
        else:
            analytics = str(response)
        
        # Get similar snags with metadata
        similar_snags = get_similar_snags_with_metadata(db, query, k=5)

        # Format as JSON
        json_results = display_results_as_json(analytics, similar_snags, query)
        
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
def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    return obj

def display_results_as_json(response_text: str, similar_snags: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """Format and display results as structured JSON"""
    num_snags = len(similar_snags)
    similarity_scores = [s['similarity_score'] for s in similar_snags]

    avg_similarity = sum(similarity_scores) / num_snags if num_snags else 0

    results = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "status": "success",
        "based_on_historical_cases": num_snags,  
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
        },
        "deep_analytics": response_text
    }

    return results

class QueryRequest(BaseModel):
    query: str
    helicopter_type: Optional[str] = None
    flight_hours: Optional[str] = None
    event_type: Optional[str] = None
    status: Optional[str] = None
    raised_by: Optional[str] = None
    
@app.post("/analytics")
async def rectification(request: QueryRequest) -> Dict[Any, Any]:
    try:
        print("🚁 Aircraft Snag Resolution System - JSON Output")
        print("=" * 50)

        chain, db = get_chain_cached()


        parts = [
            f"Query: {request.query}",
            f"Helicopter Type: {request.helicopter_type}" if request.helicopter_type else "",
            f"Flight Hours: {request.flight_hours}"  if request.flight_hours else "",
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
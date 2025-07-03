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
from fastapi.encoders import jsonable_encoder
import numpy as np
from functools import lru_cache
import json
import shutil
import ast


import re

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # ✅ exact origin (wildcards won't work with credentials)
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

        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 20}) 
        logger.info("Retriever configured successfully.")

        
        prompt = PromptTemplate.from_template('''
You are an expert aircraft technician and data analyst with deep experience in helicopter maintenance and snag analysis.

Given the current snag and historical records, perform an analytical review by:
1. Matching the current snag with similar historical records.
2. Inferring common patterns and metrics.
3. Creating dynamic analytics including categories and labels from the historical context.

---
Current Snag:
{question}

Historical Snag Records:
{context}
---

🔧 TASK:
Analyze the current snag using the matched historical records and return only structured analytics in **valid JSON format**.

INSTRUCTIONS:
- Create relevant **categories** for the pie chart based on snag types or themes.
- Identify common **event types** (e.g., GR, PLT, Ground Observation) and their no. of occurences in integer for Bar Chart 1.
- Identify common **aircraft types** or codes (e.g., IA, J, ZD, etc.) and their no. of occurences in integer for Bar Chart 2.
- Estimate snag metrics (Complexity, Time, Tools, Risk, Frequency) using a 1-5 scale.
- Do not include explanations or commentary — just valid JSON.

IMPORTANT: If the query is not related to the historical snag records, RETURN EXACTLY this JSON:
{{ "RadarChart": {{}}, "PieChart": {{}}, "BarChart1": {{}}, "BarChart2": {{}} }}

Otherwise, return a valid JSON response in the format below.
🎯 OUTPUT FORMAT:
```json
{{
  "RadarChart": {{
    "Complexity": 1-5,
    "Time Needed": 1-5,
    "Tools Required": 1-5,
    "Risk Level": 1-5,
    "Frequency": 1-5
  }},
  "PieChart": {{
    "Cat-1(insert your category)": 1-100,
    "Cat-2(insert your category)": 1-100,
    "Cat-3(insert your category)": 1-100
  }},
  "BarChart1": {{
    "PLT": int,
    "GR": int,
    "Ground Observation": int
  }},
  "BarChart2": {{
    "IA": int,
    "J": int,
    "ZD": int,
    "IN": int,
    "CG": int
  }}
}}

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
    



def get_similar_snags_with_metadata(db, query: str, k: int = 5) -> List[Dict[str, Any]]:
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
    
def clean_json_block(llm_response: str):
    try:
        # Remove ```json and ``` if present
        clean = re.sub(r"^```json\s*|\s*```$", "", llm_response.strip())
        return json.loads(clean)
    except json.JSONDecodeError as e:
        print("❌ JSON parsing failed:", e)
        return {}

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
    print(response_text)

    num_snags = len(similar_snags)
    similarity_scores = [s['similarity_score'] for s in similar_snags]
    avg_similarity = sum(similarity_scores) / num_snags if num_snags else 0
    parsed = clean_json_block(response_text)
    radar_chart = parsed.get("RadarChart", {})
    pie_chart = parsed.get("PieChart", {})
    bar_chart1 = parsed.get("BarChart1", {})
    bar_chart2 = parsed.get("BarChart2", {})
    
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
        "RadarChart": [{"category": key, "value": value} for key, value in radar_chart.items()],
        "PieChart": [{"category": key, "value": value} for key, value in pie_chart.items()],
        "BarChart1": [{"category": key, "value": value} for key, value in bar_chart1.items()],
        "BarChart2": [{"category": key, "value": value} for key, value in bar_chart2.items()]
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

        chain, db = get_chain()

        final_query = request.query 

        print("🔍 Final LLM Query:\n", final_query)

        json_results = process_snag_query_json(chain, db, final_query)
        return convert_numpy(json_results)

    except Exception as e:
        return {"error": str(e)}
    
from langchain.chains import LLMChain

@app.post("/verify")
async def rectification(request: QueryRequest) -> Dict[str, Any]:
    try:
        final_query = request.query 
        logger.info("🔍 Received query for snag verification.")

        prompt = PromptTemplate.from_template("""
        You are an expert aircraft technician and data analyst.

        Your task is to determine whether the following input is a valid aircraft snag description or just random or irrelevant text.

        A valid aircraft snag will typically describe an issue or malfunction in aircraft components or systems, often in clear technical terms.

        ---
        Input:
        {question}
        ---

        Answer with **only** one word: "Yes" if it is a valid snag description, or "No" if it is arbitrary or meaningless or inappropriate.

        Respond with just: Yes or No.
        """)

        llm_chain = LLMChain(
            llm=get_llm(),  # e.g., ChatOpenAI(temperature=0.0)
            prompt=prompt,
            verbose=True
        )

        response = llm_chain.invoke({"question": final_query})
        raw_result = response.get("text", "").strip().lower()

        # Optional sanitization
        if "yes" in raw_result:
            return {"result": "Yes"}
        elif "no" in raw_result:
            return {"result": "No"}
        else:
            return {"result": "No", "note": "Model response not clearly yes/no. Defaulting to No."}

    except Exception as e:
        logger.error(f"❌ Error in rectification: {e}")
        return {"error": str(e)}

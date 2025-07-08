from datetime import datetime
import os
import logging
from collections import defaultdict
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains import LLMChain
from services.llm import get_llm
from services.chain_service import get_analytics_chain
from services.parsers import process_snag_query_json_analysis
from utils.utils import convert_numpy
from models.models import QueryRequest

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
    
@app.post("/analytics")
async def rectification(request: QueryRequest) -> Dict[Any, Any]:
    try:
        print("🚁 Aircraft Snag Resolution System - JSON Output")
        print("=" * 50)

        chain, db = get_analytics_chain()

        final_query = request.query 

        print("🔍 Final LLM Query:\n", final_query)

        json_results = process_snag_query_json_analysis(chain, db, final_query)
        return convert_numpy(json_results)

    except Exception as e:
        return {"error": str(e)}
    

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
            llm=get_llm(), 
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

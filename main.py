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
from models.models import QueryRequestFile
from services.excel_service import excel_to_documents
from services.chain_service import get_analytics_chain_from_xls
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


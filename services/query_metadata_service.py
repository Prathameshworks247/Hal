"""
Query analysis service for inferring metadata from user queries.
Used to enhance Global FAISS retrieval with department and document_type filtering.
"""
import logging
from typing import Dict, Optional
from services.llm import get_llm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

logger = logging.getLogger(__name__)


def infer_query_metadata(query: str) -> Dict[str, Optional[str]]:
    """
    Infer department and document_type from user query using LLM.
    
    Args:
        query: User's search query
        
    Returns:
        Dictionary with 'department' and 'document_type' (None if cannot infer)
    """
    try:
        prompt = PromptTemplate.from_template("""
You are an expert aircraft maintenance query analyzer. Your task is to infer metadata from user queries.

DEPARTMENTS (choose ONE or return null):
- structures: Wing, fuselage, airframe, structural components
- avionics: Electronics, navigation, communication systems, instruments
- propulsion: Engines, fuel systems, exhaust, turbines
- hydraulics: Hydraulic systems, landing gear, brakes
- electrical: Electrical systems, wiring, batteries, generators
- null: If the query doesn't clearly fit any department

DOCUMENT TYPES (choose ONE or return null):
- manual: Maintenance manuals, technical manuals, service manuals
- training_manual: Training materials, educational documents
- inspection_report: Inspection procedures, checklists, reports
- troubleshooting_guide: Troubleshooting procedures, diagnostic guides
- null: If the query doesn't clearly specify a document type

---

USER QUERY:
{query}

---

INSTRUCTIONS:
1. Analyze the query to determine which department it relates to
2. Determine what type of document the user is likely looking for
3. Only return a value if you are CONFIDENT (>80% sure)
4. Return "null" if uncertain or if the query is too general

RESPOND IN THIS EXACT FORMAT (valid JSON only):
{{
  "department": "structures" or "avionics" or "propulsion" or "hydraulics" or "electrical" or null,
  "document_type": "manual" or "training_manual" or "inspection_report" or "troubleshooting_guide" or null,
  "confidence": "high" or "medium" or "low"
}}

Do not include any explanation, just the JSON.
""")
        
        llm = get_llm()
        chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
        
        response = chain.invoke({"query": query})
        result_text = response.get("text", "").strip()
        
        # Parse JSON response
        import json
        # Remove markdown code blocks if present
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        parsed = json.loads(result_text)
        
        # Extract values, converting "null" strings to None
        department = parsed.get("department")
        document_type = parsed.get("document_type")
        confidence = parsed.get("confidence", "low")
        
        if department == "null":
            department = None
        if document_type == "null":
            document_type = None
        
        logger.info(f"Query metadata inference - dept: {department}, type: {document_type}, confidence: {confidence}")
        
        return {
            "department": department,
            "document_type": document_type,
            "confidence": confidence
        }
        
    except Exception as e:
        logger.warning(f"Failed to infer query metadata: {str(e)}")
        return {
            "department": None,
            "document_type": None,
            "confidence": "none"
        }


def create_metadata_filter(department: Optional[str] = None, document_type: Optional[str] = None) -> Optional[Dict]:
    """
    Create a metadata filter for FAISS retrieval.
    
    Args:
        department: Department to filter by
        document_type: Document type to filter by
        
    Returns:
        Filter dictionary or None if no filters
    """
    if not department and not document_type:
        return None
    
    filter_dict = {}
    if department:
        filter_dict["department"] = department
    if document_type:
        filter_dict["document_type"] = document_type
    
    return filter_dict

"""
Query analysis service for inferring metadata from user queries.
Used to enhance Global FAISS retrieval with department filtering.
"""
import logging
from typing import Dict, Optional
from services.llm import get_llm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

logger = logging.getLogger(__name__)


def infer_query_metadata(query: str) -> Dict[str, Optional[str]]:
    """
    Infer department from user query using LLM.
    
    Args:
        query: User's search query
        
    Returns:
        Dictionary with 'department' (None if cannot infer)
    """
    try:
        prompt = PromptTemplate.from_template("""
You are an expert aircraft maintenance query analyzer. Your task is to infer metadata from user queries.

DEPARTMENTS (choose ONE or return null):
- structures: Wing, fuselage, airframe, structural components
- avionics: Electronics, navigation, communication systems, instruments
- propulsion: Engines, fuel systems, exhaust, turbines
- maintenance: General maintenance, service manuals, scheduling
- general: if the query fits general aviation or multiple categories
- null: If the query doesn't clearly fit any department

---

USER QUERY:
{query}

---

INSTRUCTIONS:
1. Analyze the query to determine which department it relates to
2. Only return a value if you are CONFIDENT (>80% sure)
3. Return "null" if uncertain or if the query is too general

RESPOND IN THIS EXACT FORMAT (valid JSON only):
{{
  "department": "structures" or "avionics" or "propulsion" or "maintenance" or "general" or null,
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
        confidence = parsed.get("confidence", "low")
        
        if department == "null":
            department = None
        
        logger.info(f"Query metadata inference - dept: {department}, confidence: {confidence}")
        
        return {
            "department": department,
            "confidence": confidence
        }
        
    except Exception as e:
        logger.warning(f"Failed to infer query metadata: {str(e)}")
        return {
            "department": None,
            "confidence": "none"
        }


def create_metadata_filter(department: Optional[str] = None) -> Optional[Dict]:
    """
    Create a metadata filter for FAISS retrieval.
    
    Args:
        department: Department to filter by
        
    Returns:
        Filter dictionary or None if no filters
    """
    if not department:
        return None
    
    filter_dict = {}
    if department:
        filter_dict["department"] = department
    
    return filter_dict

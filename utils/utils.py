"""
Utility functions for the Sky-Sentinal application.
"""
import numpy as np
from services.similarity_service import get_similar_snags_with_metadata
import json
import re


def convert_numpy(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    return obj


def test_retriever(db, query):
    """
    Test function to check if retriever is working.
    
    Args:
        db: FAISS vectorstore instance
        query: Test query string
        
    Returns:
        True if retriever works, False otherwise
    """
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


def clean_json_block(llm_response: str):
    """
    Clean and parse JSON from LLM response string.
    Removes markdown code block markers if present.
    
    Args:
        llm_response: Raw LLM response string (may contain ```json markers)
        
    Returns:
        Parsed JSON object, or empty dict if parsing fails
    """
    try:
        # Remove ```json and ``` if present
        clean = re.sub(r"^```json\s*|\s*```$", "", llm_response.strip())
        return json.loads(clean)
    except json.JSONDecodeError as e:
        print("❌ JSON parsing failed:", e)
        return {}


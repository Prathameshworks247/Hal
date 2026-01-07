import logging
from typing import Dict,Any,List
from services.similarity_service import get_similar_snags_with_metadata, get_similar_records_with_metadata
from datetime import datetime
from services.similarity_service import get_similar_snags_analysis
from utils.utils import  clean_json_block
from services.pdf_citation_service import extract_citations_from_retrieved_docs, store_citations
import uuid

logger = logging.getLogger(__name__)

def process_snag_query_json(chain, db, query: str) -> Dict[str, Any]:
    try:
        logger.info(f"Processing query: {query}")
        
        # Get AI-generated rectification
        response = chain.invoke({"question": query})
        
        # Extract result and source documents
        if isinstance(response, dict):
            rectification = response.get('result', response.get('answer', str(response)))
            source_documents = response.get('source_documents', [])
        else:
            rectification = str(response)
            source_documents = []
        
        # Get similar snags with metadata
        similar_snags = get_similar_snags_with_metadata(db, query, k=5)
        
        # Extract PDF citations with bbox coordinates
        # Use source_documents if available, otherwise extract from similar_snags
        docs_for_citation = source_documents if source_documents else []
        
        # If no source_documents, extract document objects from similar_snags
        if not docs_for_citation and similar_snags:
            docs_for_citation = [
                snag['document'] 
                for snag in similar_snags 
                if 'document' in snag and snag.get('metadata', {}).get('file_type') == 'pdf'
            ]
        
        logger.info(f"Extracting PDF citations from {len(docs_for_citation)} documents")
        pdf_citations = extract_citations_from_retrieved_docs(docs_for_citation, query)
        logger.info(f"Extracted {len(pdf_citations)} PDF citations")
        
        # Generate session ID for citation storage
        session_id = str(uuid.uuid4())
        store_citations(session_id, pdf_citations)

        # Format as JSON
        json_results = display_results_as_json(rectification, similar_snags, query, pdf_citations, session_id)
        
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


def display_results_as_json(response_text: str, similar_snags: List[Dict[str, Any]], query: str, 
                           citations: List[Dict[str, Any]] = None, session_id: str = None) -> Dict[str, Any]:
    """Format and display results as structured JSON with PDF citations"""
    num_snags = len(similar_snags)
    
    # Remove 'document' key from similar_snags (not JSON serializable)
    clean_snags = []
    for snag in similar_snags:
        clean_snag = {k: v for k, v in snag.items() if k != 'document'}
        clean_snags.append(clean_snag)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "status": "success",
        "session_id": session_id,
        "rectification": {
            "ai_recommendation": response_text,
            "based_on_historical_cases": num_snags,
            "confidence": "high" if num_snags >= 3 else "medium" if num_snags >= 2 else "low"
        },
        "similar_historical_snags": clean_snags,
        "citations": citations if citations else []
    }

    return results

def process_file_query_json(chain, db, query: str) -> Dict[str, Any]:
    """
    Process file-specific query and return results with PDF citations
    
    Args:
        chain: QA chain instance
        db: FAISS database instance
        query: User query
        
    Returns:
        Complete JSON response with rectification, similar snags, and citations
    """
    try:
        logger.info(f"Processing file query: {query}")
        
        # Get AI-generated rectification
        response = chain.invoke({"question": query})
        
        # Extract result and source documents
        if isinstance(response, dict):
            rectification = response.get('result', response.get('answer', str(response)))
            source_documents = response.get('source_documents', [])
        else:
            rectification = str(response)
            source_documents = []
        
        # Get similar snags with metadata
        similar_snags = get_similar_records_with_metadata(db, query, k=5)
        
        # Extract PDF citations with bbox coordinates
        # Use source_documents if available, otherwise extract from similar_snags
        docs_for_citation = source_documents if source_documents else []
        
        # If no source_documents, extract document objects from similar_snags
        if not docs_for_citation and similar_snags:
            docs_for_citation = [
                snag['document'] 
                for snag in similar_snags 
                if 'document' in snag and snag.get('metadata', {}).get('file_type') == 'pdf'
            ]
        
        logger.info(f"Extracting PDF citations from {len(docs_for_citation)} documents")
        pdf_citations = extract_citations_from_retrieved_docs(docs_for_citation, query)
        logger.info(f"Extracted {len(pdf_citations)} PDF citations")
        
        # Generate session ID for citation storage
        session_id = str(uuid.uuid4())
        store_citations(session_id, pdf_citations)

        # Format as JSON
        json_results = display_results_as_json(rectification, similar_snags, query, pdf_citations, session_id)
        
        return json_results
        
    except Exception as e:
        logger.error(f"Error processing file query: {str(e)}")
        from datetime import datetime
        return {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "status": "error",
            "error_message": str(e),
            "session_id": None,
            "rectification": {
                "ai_recommendation": None,
                "based_on_historical_cases": 0,
                "confidence": "none"
            },
            "similar_historical_snags": [],
            "citations": []
        }


def process_snag_query_json_analysis(chain, db, query: str) -> Dict[str, Any]:
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
        similar_snags = get_similar_snags_analysis(db, query)

        # Format as JSON
        json_results = display_results_as_json_analysis(analytics, similar_snags, query)
        
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



def display_results_as_json_analysis(response_text: str, similar_snags: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """Format and display results as structured JSON"""
    print(response_text)

    num_snags = len(similar_snags)
    similarity_scores = [s['similarity_score'] for s in similar_snags]
    avg_similarity = sum(similarity_scores) / num_snags if num_snags else 0
    parsed = clean_json_block(response_text)
    radar_chart = parsed.get("RadarChart", {})
    pie_chart = parsed.get("PieChart", {})
    bar_chart1 = parsed.get("BarChart1", {})
    bar_chart2 = parsed["BarChart2"] if "BarChart2" in parsed and parsed["BarChart2"] else None
    
    results = {
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
        "graphs":{
            "RadarChart": [{"category": key, "value": value} for key, value in radar_chart.items()],
            "PieChart": [{"category": key, "value": value} for key, value in pie_chart.items()],
            "BarChart1": [{"category": key, "value": value} for key, value in bar_chart1.items()],
            "BarChart2": [{"category": key, "value": value} for key, value in bar_chart2.items()] if bar_chart2 else [],
        }
       
    }

    return results

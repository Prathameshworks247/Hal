import logger
from typing import Dict,Any,List
from services.similarity_service import get_similar_snags_with_metadata
from datetime import datetime
from services.similarity_service import get_similar_snags_analysis
from utils.utils import  clean_json_block

def process_snag_query_json(chain, db, query: str) -> Dict[str, Any]:
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


def display_results_as_json(response_text: str, similar_snags: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """Format and display results as structured JSON"""
    num_snags = len(similar_snags)
    results = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "status": "success",
        "rectification": {
            "ai_recommendation": response_text,
            "based_on_historical_cases": num_snags
        },
        "similar_historical_snags": similar_snags,

    }

    return results

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
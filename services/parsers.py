import logging
from typing import Dict,Any,List, Optional
from services.similarity_service import get_similar_snags_with_metadata, get_similar_records_with_metadata
from datetime import datetime
from services.similarity_service import get_similar_snags_analysis
from utils.utils import  clean_json_block
from services.pdf_citation_service import extract_citations_from_retrieved_docs, store_citations
from services.session_faiss_manager import SessionFAISSManager
from langchain.schema import Document
import uuid

logger = logging.getLogger(__name__)

def process_snag_query_json(
    chain, 
    db, 
    search_query: str, 
    user_query: str = None, 
    conversation_context: dict = None,
    session_manager: Optional[SessionFAISSManager] = None,
    citation_session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process snag query with optional conversation context and session awareness.
    
    Args:
        chain: QA chain
        db: Vector database (GLOBAL_FAISS)
        search_query: Query optimized for RAG retrieval
        user_query: Original user query (for display)
        conversation_context: Conversation history context
        session_manager: Optional SessionFAISSManager for session-specific retrieval
        citation_session_id: Session ID for citation storage
    """
    try:
        # Use search_query for retrieval, user_query for display
        if user_query is None:
            user_query = search_query
            
        logger.info(f"Processing query: {user_query}")
        if session_manager:
            logger.info(f"  Session: {session_manager.session_id}, Has file: {session_manager.has_uploaded_file()}")
        if conversation_context and conversation_context.get('has_context'):
            logger.info(f"  With conversation context: {conversation_context.get('context_summary')}")
        
        # Build enhanced query with conversation context
        if conversation_context and conversation_context.get('has_context'):
            # Prepend conversation context to help LLM understand references
            context_prefix = f"[Conversation Context: {conversation_context['full_context']}]\n\nCurrent Question: "
            llm_query = context_prefix + user_query
        else:
            llm_query = user_query
        
        # Determine retrieval strategy based on session
        retrieved_docs: List[Document] = []
        
        if session_manager and session_manager.has_uploaded_file():
            # CASE 1: User uploaded file - retrieve ONLY from SESSION_FAISS
            logger.info("Retrieving from SESSION_FAISS only (user uploaded file)")
            retrieved_docs = session_manager.retrieve_from_session(search_query, k=5)
            
        elif session_manager:
            # CASE 2: No uploaded file - retrieve from BOTH (global + session memory)
            logger.info("Retrieving from GLOBAL_FAISS + SESSION_FAISS (conversation memory)")
            
            # Get from global FAISS
            global_results = db.similarity_search(search_query, k=3)
            
            # Get from session FAISS (conversation memory)
            session_results = session_manager.retrieve_from_session(search_query, k=2)
            
            # Merge with authority rules
            retrieved_docs = SessionFAISSManager.merge_retrieval_results(
                global_results=global_results,
                session_results=session_results,
                k=5,
                prioritize="documents"
            )
            
        else:
            # CASE 3: No session - retrieve from GLOBAL_FAISS only (legacy behavior)
            logger.info("Retrieving from GLOBAL_FAISS only (no session)")
            retrieved_docs = db.similarity_search(search_query, k=5)
        
        # Get AI-generated rectification
        response = chain.invoke({"question": llm_query})
        
        # Extract result and source documents
        if isinstance(response, dict):
            rectification = response.get('result', response.get('answer', str(response)))
            source_documents = response.get('source_documents', [])
        else:
            rectification = str(response)
            source_documents = []
        
        # Use retrieved_docs if no source_documents from chain
        if not source_documents:
            source_documents = retrieved_docs
        
        # Get similar snags with metadata (use search_query for better retrieval)
        similar_snags = get_similar_snags_with_metadata(db, search_query, k=5)
        
        # Extract PDF citations with bbox coordinates
        docs_for_citation = source_documents if source_documents else []
        
        # If no source_documents, extract document objects from similar_snags
        if not docs_for_citation and similar_snags:
            docs_for_citation = [
                snag['document'] 
                for snag in similar_snags 
                if 'document' in snag and snag.get('metadata', {}).get('file_type') == 'pdf'
            ]
        
        logger.info(f"Extracting PDF citations from {len(docs_for_citation)} documents")
        pdf_citations = extract_citations_from_retrieved_docs(docs_for_citation, user_query)
        logger.info(f"Extracted {len(pdf_citations)} PDF citations")
        
        # Use provided citation_session_id or generate new
        citation_sid = citation_session_id or str(uuid.uuid4())
        store_citations(citation_sid, pdf_citations)

        # Format as JSON (use user_query for display)
        json_results = display_results_as_json(rectification, similar_snags, user_query, pdf_citations, citation_sid)
        
        # Add conversation context to response if available
        if conversation_context and conversation_context.get('has_context'):
            json_results["conversation_context"] = {
                "has_history": True,
                "history_length": conversation_context.get('history_length', 0),
                "context_used": conversation_context.get('context_summary')
            }
        else:
            json_results["conversation_context"] = {
                "has_history": False
            }
        
        return json_results, rectification  # Return rectification for conversation memory
        
    except Exception as e:
        logger.error(f"Error processing snag query: {str(e)}")
        from datetime import datetime
        return {
            "timestamp": datetime.now().isoformat(),
            "query": user_query if user_query else search_query,
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
        }, None


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

def process_file_query_json(
    chain, 
    db, 
    search_query: str, 
    user_query: str = None, 
    conversation_context: dict = None,
    session_manager: Optional[SessionFAISSManager] = None,
    citation_session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process file-specific query and return results with PDF citations
    
    Args:
        chain: QA chain instance
        db: FAISS database instance
        search_query: Query optimized for RAG retrieval
        user_query: Original user query (for display)
        conversation_context: Conversation history context
        session_manager: Optional SessionFAISSManager for session-specific retrieval
        citation_session_id: Session ID for citation storage
        
    Returns:
        Complete JSON response with rectification, similar snags, and citations
    """
    try:
        if user_query is None:
            user_query = search_query
            
        logger.info(f"Processing file query: {user_query}")
        if session_manager:
            logger.info(f"  Session: {session_manager.session_id}, Has file: {session_manager.has_uploaded_file()}")
        if conversation_context and conversation_context.get('has_context'):
            logger.info(f"  With conversation context: {conversation_context.get('context_summary')}")
        
        # Build enhanced query with conversation context
        if conversation_context and conversation_context.get('has_context'):
            context_prefix = f"[Conversation Context: {conversation_context['full_context']}]\n\nCurrent Question: "
            llm_query = context_prefix + user_query
        else:
            llm_query = user_query
        
        # Determine retrieval strategy based on session
        retrieved_docs: List[Document] = []
        
        if session_manager and session_manager.has_uploaded_file():
            # CASE 1: User uploaded file - retrieve ONLY from SESSION_FAISS
            logger.info("Retrieving from SESSION_FAISS only (user uploaded file)")
            retrieved_docs = session_manager.retrieve_from_session(search_query, k=5)
            
        elif session_manager:
            # CASE 2: No uploaded file - retrieve from BOTH (global + session memory)
            logger.info("Retrieving from GLOBAL_FAISS + SESSION_FAISS (conversation memory)")
            
            # Get from file-specific FAISS
            file_results = db.similarity_search(search_query, k=3)
            
            # Get from session FAISS (conversation memory)
            session_results = session_manager.retrieve_from_session(search_query, k=2)
            
            # Merge with authority rules
            retrieved_docs = SessionFAISSManager.merge_retrieval_results(
                global_results=file_results,
                session_results=session_results,
                k=5,
                prioritize="documents"
            )
            
        else:
            # CASE 3: No session - retrieve from file FAISS only (legacy behavior)
            logger.info("Retrieving from file FAISS only (no session)")
            retrieved_docs = db.similarity_search(search_query, k=5)
        
        # Get AI-generated rectification (use llm_query with context)
        response = chain.invoke({"question": llm_query})
        
        # Extract result and source documents
        if isinstance(response, dict):
            rectification = response.get('result', response.get('answer', str(response)))
            source_documents = response.get('source_documents', [])
        else:
            rectification = str(response)
            source_documents = []
        
        # Use retrieved_docs if no source_documents from chain
        if not source_documents:
            source_documents = retrieved_docs
        
        # Get similar snags with metadata (use search_query for retrieval)
        similar_snags = get_similar_records_with_metadata(db, search_query, k=5)
        
        # Extract PDF citations with bbox coordinates
        docs_for_citation = source_documents if source_documents else []
        
        # If no source_documents, extract document objects from similar_snags
        if not docs_for_citation and similar_snags:
            docs_for_citation = [
                snag['document'] 
                for snag in similar_snags 
                if 'document' in snag and snag.get('metadata', {}).get('file_type') == 'pdf'
            ]
        
        logger.info(f"Extracting PDF citations from {len(docs_for_citation)} documents")
        pdf_citations = extract_citations_from_retrieved_docs(docs_for_citation, user_query)
        logger.info(f"Extracted {len(pdf_citations)} PDF citations")
        
        # Use provided citation_session_id or generate new
        citation_sid = citation_session_id or str(uuid.uuid4())
        store_citations(citation_sid, pdf_citations)

        # Format as JSON (use user_query for display)
        json_results = display_results_as_json(rectification, similar_snags, user_query, pdf_citations, citation_sid)
        
        # Add conversation context to response if available
        if conversation_context and conversation_context.get('has_context'):
            json_results["conversation_context"] = {
                "has_history": True,
                "history_length": conversation_context.get('history_length', 0),
                "context_used": conversation_context.get('context_summary')
            }
        else:
            json_results["conversation_context"] = {
                "has_history": False
            }
        
        return json_results, rectification  # Return rectification for conversation memory
        
    except Exception as e:
        logger.error(f"Error processing file query: {str(e)}")
        from datetime import datetime
        return {
            "timestamp": datetime.now().isoformat(),
            "query": user_query if user_query else search_query,
            "status": "error",
            "error_message": str(e),
            "session_id": citation_session_id,
            "rectification": {
                "ai_recommendation": None,
                "based_on_historical_cases": 0,
                "confidence": "none"
            },
            "similar_historical_snags": [],
            "citations": []
        }, None


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

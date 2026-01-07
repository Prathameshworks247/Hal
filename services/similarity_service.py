import re
from typing import List, Dict,Any
import spacy

nlp = spacy.load("en_core_web_md")  

def has_semantic_meaning(sentence):
    doc = nlp(sentence)
    return doc.vector_norm > 0.3  # tweak threshold if needed


def get_similar_snags_with_metadata(db, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve similar snags with their metadata and normalized similarity scores.
    """
    try:
        docs_with_scores = db.similarity_search_with_score(query, k=k)

        similar_snags = []
        for i, (doc, score) in enumerate(docs_with_scores):
            if score > 0.9:
                # similar_snags.append({
                #      'rank': "none",
                #     'fields': "none",
                #     'metadata': "none"
                # })
                continue  
            key_values = extract_key_values(doc.page_content)
            
            record = {
                'rank': i + 1,
                'fields': key_values,
                'metadata': doc.metadata,
                'document': doc  # Add the actual document object for citation extraction
            }
            similar_snags.append(record)

        return similar_snags

    except Exception as e:
        logger.error(f"Error retrieving similar snags: {str(e)}")
        return []



def extract_key_values(text: str) -> Dict[str, str]:
    """
    Extracts key-value pairs from a semi-structured text.
    Each line is expected to be in the form `KEY: VALUE`
    """
    result = {}
    lines = text.split('\n')
    for line in lines:
        match = re.match(r"^\s*(.+?):\s*(.+)", line)
        if match:
            key, value = match.groups()
            result[key.strip().lower()] = value.strip()
    return result

def get_similar_records_with_metadata(db, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve similar records from a vector DB with metadata and similarity scores (Excel agnostic).
    
    Args:
        db: FAISS or other vectorstore instance
        query: Text query for similarity search
        k: Number of top matches to retrieve
    
    Returns:
        List of dictionaries with extracted key-values, metadata, and similarity info
    """
    try:
        docs_with_scores = db.similarity_search_with_score(query, k=k)

        results = []
        for i, (doc, score) in enumerate(docs_with_scores):
            key_values = extract_key_values(doc.page_content)
            if score>0.9:
                continue

            record = {
                'rank': i + 1,
                'fields': key_values,
                'metadata': doc.metadata,
                'document': doc  # Add the actual document object for citation extraction
            }

            results.append(record)

        return results

    except Exception as e:
        logger.error(f"Error retrieving similar records: {str(e)}")
        return []

def get_similar_snags_analysis(db, query: str) -> List[Dict[str, Any]]:
    try:
        # Get similar documents with scores
        max_k = len(db.docstore._dict)

        results = db.similarity_search_with_score(query, k=max_k)

        docs_with_scores = [
            (doc, score) for doc, score in results if score > 0.9
        ]

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
                'similarity_percentage': round((score) * 100, 2)
            }

            similar_snags.append(snag_info)

        return similar_snags

    except Exception as e:
        logger.error(f"Error retrieving similar snags: {str(e)}")
        return []
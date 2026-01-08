"""
Hybrid retrieval service combining semantic search (FAISS) with BM25 keyword search.
Includes cross-encoder reranking for improved precision.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
import re

logger = logging.getLogger(__name__)

# BM25 imports
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank-bm25 not available. Install: pip install rank-bm25")

# Cross-encoder imports
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.warning("sentence-transformers not available for reranking. Install: pip install sentence-transformers")


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text for BM25. Simple tokenization that works without NLTK.
    Falls back to basic word splitting if NLTK not available.
    """
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            # Download punkt tokenizer if not available
            nltk.download('punkt', quiet=True)
        
        from nltk.tokenize import word_tokenize
        # Tokenize and lowercase
        tokens = word_tokenize(text.lower())
        # Remove punctuation-only tokens and short tokens
        tokens = [t for t in tokens if t.isalnum() and len(t) > 1]
        return tokens
    except (ImportError, Exception) as e:
        # Fallback to simple tokenization
        logger.debug(f"NLTK tokenization not available, using simple tokenizer: {e}")
        # Simple tokenization: lowercase, split on whitespace and punctuation
        text_lower = text.lower()
        tokens = re.findall(r'\b\w+\b', text_lower)
        tokens = [t for t in tokens if len(t) > 1]
        return tokens


class HybridRetriever:
    """
    Hybrid retriever combining semantic (FAISS) and keyword (BM25) search.
    """
    
    def __init__(self, faiss_db, documents: Optional[List[Document]] = None):
        """
        Initialize hybrid retriever.
        
        Args:
            faiss_db: FAISS vectorstore instance
            documents: Optional list of documents for BM25 indexing. 
                      If None, will build from FAISS on first use.
        """
        self.faiss_db = faiss_db
        self.bm25 = None
        self.documents = documents or []
        self.bm25_initialized = False
        
        # Initialize cross-encoder for reranking (lazy loading)
        self.cross_encoder = None
        self.cross_encoder_initialized = False
        
    def _initialize_bm25(self):
        """Initialize BM25 index from documents."""
        if self.bm25_initialized:
            return
            
        if not BM25_AVAILABLE:
            logger.warning("BM25 not available, using semantic search only")
            self.bm25_initialized = True
            return
        
        try:
            # Get documents if not provided
            if not self.documents:
                # Try to get documents from FAISS
                # Search with a very broad query to get all documents
                try:
                    # Get a sample query to retrieve documents
                    sample_docs = self.faiss_db.similarity_search("", k=1000)
                    self.documents = sample_docs
                except Exception as e:
                    logger.warning(f"Could not retrieve documents from FAISS for BM25: {e}")
                    self.documents = []
            
            if not self.documents:
                logger.warning("No documents available for BM25 indexing")
                self.bm25_initialized = True
                return
            
            # Tokenize all documents
            tokenized_docs = []
            for doc in self.documents:
                tokens = tokenize_text(doc.page_content)
                tokenized_docs.append(tokens)
            
            # Initialize BM25
            self.bm25 = BM25Okapi(tokenized_docs)
            self.bm25_initialized = True
            logger.info(f"BM25 initialized with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Error initializing BM25: {e}")
            self.bm25_initialized = True  # Mark as initialized to avoid retrying
    
    def _initialize_cross_encoder(self):
        """Initialize cross-encoder for reranking."""
        if self.cross_encoder_initialized:
            return
            
        if not CROSS_ENCODER_AVAILABLE:
            logger.debug("Cross-encoder not available, skipping reranking")
            self.cross_encoder_initialized = True
            return
        
        try:
            # Use a lightweight cross-encoder model
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            logger.info(f"Loading cross-encoder model: {model_name}")
            self.cross_encoder = CrossEncoder(model_name)
            self.cross_encoder_initialized = True
            logger.info("Cross-encoder initialized successfully")
        except Exception as e:
            logger.warning(f"Error initializing cross-encoder: {e}")
            self.cross_encoder_initialized = True  # Mark as initialized to avoid retrying
    
    def hybrid_search(
        self, 
        query: str, 
        k: int = 5,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
        rerank: bool = True,
        rerank_top_k: int = 20
    ) -> List[Document]:
        """
        Perform hybrid search combining semantic and keyword search.
        
        Args:
            query: Search query
            k: Number of results to return (after reranking)
            semantic_weight: Weight for semantic search scores (default: 0.6)
            keyword_weight: Weight for BM25 keyword search scores (default: 0.4)
            rerank: Whether to use cross-encoder reranking (default: True)
            rerank_top_k: Number of candidates to rerank (default: 20)
            
        Returns:
            List of retrieved documents
        """
        # Ensure BM25 is initialized
        self._initialize_bm25()
        
        # Get semantic search results
        try:
            # Get more results for better reranking
            semantic_docs_with_scores = self.faiss_db.similarity_search_with_score(query, k=rerank_top_k if rerank else k)
            semantic_scores = {i: (doc, score) for i, (doc, score) in enumerate(semantic_docs_with_scores)}
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            semantic_scores = {}
        
        # Get BM25 keyword search results
        keyword_scores = {}
        if self.bm25 and BM25_AVAILABLE:
            try:
                query_tokens = tokenize_text(query)
                if query_tokens:
                    # Get BM25 scores for all documents
                    bm25_scores = self.bm25.get_scores(query_tokens)
                    
                    # Normalize BM25 scores (0-1 range)
                    if len(bm25_scores) > 0:
                        max_score = max(bm25_scores)
                        min_score = min(bm25_scores)
                        if max_score > min_score:
                            bm25_scores_normalized = [(s - min_score) / (max_score - min_score) for s in bm25_scores]
                        else:
                            bm25_scores_normalized = [0.5] * len(bm25_scores)
                        
                        # Map to documents (assuming documents match semantic search docs)
                        for i, (doc, _) in enumerate(semantic_docs_with_scores):
                            if i < len(bm25_scores_normalized):
                                keyword_scores[i] = bm25_scores_normalized[i]
            except Exception as e:
                logger.warning(f"Error in BM25 search: {e}")
        
        # Combine scores
        combined_scores = {}
        for idx, (doc, semantic_score) in semantic_scores.items():
            # Convert FAISS distance to similarity (lower distance = higher similarity)
            # FAISS returns L2 distance, convert to similarity score (0-1)
            semantic_sim = 1 / (1 + semantic_score) if semantic_score > 0 else 1.0
            
            # Get keyword score (default to 0 if not available)
            keyword_score = keyword_scores.get(idx, 0.0)
            
            # Combine scores
            combined_score = (semantic_weight * semantic_sim) + (keyword_weight * keyword_score)
            combined_scores[idx] = (doc, combined_score)
        
        # Sort by combined score (descending)
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1][1], reverse=True)
        
        # Get top candidates for reranking or final results
        if rerank and len(sorted_results) > k:
            # Take top rerank_top_k for reranking
            candidates = sorted_results[:rerank_top_k]
        else:
            candidates = sorted_results[:k]
        
        # Rerank with cross-encoder if enabled
        if rerank and len(candidates) > 0:
            reranked_results = self._rerank_with_cross_encoder(query, candidates)
            # Take top k after reranking
            final_results = reranked_results[:k]
        else:
            final_results = candidates
        
        # Extract documents
        retrieved_docs = [doc for _, (doc, _) in final_results]
        
        logger.debug(f"Hybrid search: Retrieved {len(retrieved_docs)} documents (semantic_weight={semantic_weight}, keyword_weight={keyword_weight}, rerank={rerank})")
        return retrieved_docs
    
    def _rerank_with_cross_encoder(self, query: str, candidates: List[Tuple[int, Tuple[Document, float]]]) -> List[Tuple[int, Tuple[Document, float]]]:
        """
        Rerank candidates using cross-encoder for better precision.
        
        Args:
            query: Search query
            candidates: List of (idx, (doc, score)) tuples
            
        Returns:
            Reranked list of candidates
        """
        # Initialize cross-encoder if needed
        self._initialize_cross_encoder()
        
        if not self.cross_encoder or not CROSS_ENCODER_AVAILABLE:
            # Return original candidates if reranking not available
            return candidates
        
        try:
            # Prepare pairs for cross-encoder: (query, document_text)
            pairs = []
            for idx, (doc, _) in candidates:
                pairs.append((query, doc.page_content))
            
            # Get reranking scores
            rerank_scores = self.cross_encoder.predict(pairs)
            
            # Create new list with reranked scores
            reranked = []
            for i, (idx, (doc, original_score)) in enumerate(candidates):
                rerank_score = float(rerank_scores[i])
                # Combine original score (20%) with rerank score (80%) for final ranking
                final_score = (0.2 * original_score) + (0.8 * rerank_score)
                reranked.append((idx, (doc, final_score)))
            
            # Sort by final score (descending)
            reranked.sort(key=lambda x: x[1][1], reverse=True)
            
            logger.debug(f"Reranked {len(reranked)} candidates using cross-encoder")
            return reranked
            
        except Exception as e:
            logger.warning(f"Error during reranking: {e}, returning original candidates")
            return candidates


def hybrid_search_with_faiss(
    faiss_db,
    query: str,
    k: int = 5,
    semantic_weight: float = 0.6,
    keyword_weight: float = 0.4,
    rerank: bool = True
) -> List[Document]:
    """
    Convenience function for hybrid search with FAISS database.
    
    Args:
        faiss_db: FAISS vectorstore instance
        query: Search query
        k: Number of results to return
        semantic_weight: Weight for semantic search (default: 0.6)
        keyword_weight: Weight for keyword search (default: 0.4)
        rerank: Whether to rerank results (default: True)
        
    Returns:
        List of retrieved documents
    """
    retriever = HybridRetriever(faiss_db)
    return retriever.hybrid_search(
        query=query,
        k=k,
        semantic_weight=semantic_weight,
        keyword_weight=keyword_weight,
        rerank=rerank
    )


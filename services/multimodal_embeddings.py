"""
Multimodal embedding service for text and image descriptions.
Handles embedding generation for both authoritative text and non-authoritative image descriptions.
"""
import logging
from typing import List, Optional
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class MultimodalEmbeddingManager:
    """
    Manages embeddings for multimodal documents (text + image descriptions).
    
    Current Strategy:
    - Text chunks: Use all-MiniLM-L6-v2 embeddings (authoritative)
    - Image descriptions: Use same text embeddings (non-authoritative)
    
    This approach allows:
    - Unified embedding space (no need for projection or fusion)
    - Single FAISS index for both text and image descriptions
    - Metadata filtering to distinguish authoritative vs non-authoritative content
    - Retrieval returns both text and image descriptions naturally
    
    Future Enhancement Options:
    - Add vision embeddings (CLIP) for actual images
    - Separate indices for text and images
    - Cross-modal retrieval with score fusion
    """
    
    def __init__(self, model_path: str = "nomic-ai/nomic-embed-text-v1.5", device: str = "cpu"):
        """
        Initialize the multimodal embedding manager.
        
        Uses nomic-embed-text-v1.5 by default (stronger than MiniLM, offline-capable).
        Alternative: "BAAI/bge-small-en-v1.5"
        
        Args:
            model_path: HuggingFace model path or local path to the text embedding model
            device: Device to run embeddings on ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.device = device
        self.embeddings = None
        
    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Get or create the embeddings model.
        
        Uses nomic-embed-text-v1.5 or bge-small-en-v1.5 for better quality.
        Models are loaded from HuggingFace cache (offline-capable).
        
        Returns:
            HuggingFaceEmbeddings instance
        """
        if self.embeddings is None:
            logger.info(f"Loading text embedding model: {self.model_path}")
            # Use trust_remote_code for models that require it (like nomic)
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_path,
                    model_kwargs={
                        'device': self.device,
                        'trust_remote_code': True  # Required for nomic-embed-text-v1.5
                    },
                    encode_kwargs={
                        'normalize_embeddings': True  # Normalize for better similarity search
                    }
                )
                logger.info(f"✓ Text embedding model loaded successfully: {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load {self.model_path}, falling back to bge-small-en-v1.5: {e}")
                # Fallback to bge-small-en-v1.5
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-small-en-v1.5",
                    model_kwargs={'device': self.device},
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info("✓ Fallback embedding model loaded: BAAI/bge-small-en-v1.5")
        return self.embeddings
    
    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """
        Embed a list of documents (both text and image descriptions).
        
        Note: This function doesn't modify the documents - FAISS handles
        embedding generation internally. This is a pass-through function
        that validates the document structure.
        
        Args:
            documents: List of Document objects
        
        Returns:
            Same list of documents (unmodified)
        """
        text_count = sum(1 for d in documents if d.metadata.get("type") != "image_description")
        image_count = sum(1 for d in documents if d.metadata.get("type") == "image_description")
        
        logger.info(f"Processing {len(documents)} documents: {text_count} text chunks, {image_count} image descriptions")
        
        # Validate document structure
        for idx, doc in enumerate(documents):
            if not doc.page_content:
                logger.warning(f"Document {idx} has empty content")
            
            if doc.metadata.get("type") == "image_description":
                # Validate image description metadata
                required_fields = ["authoritative", "confidence", "page_number"]
                missing_fields = [f for f in required_fields if f not in doc.metadata]
                if missing_fields:
                    logger.warning(f"Image description missing metadata: {missing_fields}")
        
        return documents
    
    def filter_by_authority(self, documents: List[Document], authoritative_only: bool = False) -> List[Document]:
        """
        Filter documents by authoritative status.
        
        Args:
            documents: List of Document objects
            authoritative_only: If True, return only authoritative documents (text chunks)
        
        Returns:
            Filtered list of documents
        """
        if not authoritative_only:
            return documents
        
        filtered = [d for d in documents if d.metadata.get("authoritative", True)]
        logger.info(f"Filtered {len(filtered)} authoritative documents from {len(documents)} total")
        return filtered
    
    def separate_by_type(self, documents: List[Document]) -> tuple:
        """
        Separate documents into text chunks and image descriptions.
        
        Args:
            documents: List of Document objects
        
        Returns:
            Tuple of (text_documents, image_description_documents)
        """
        text_docs = []
        image_docs = []
        
        for doc in documents:
            if doc.metadata.get("type") == "image_description":
                image_docs.append(doc)
            else:
                text_docs.append(doc)
        
        logger.info(f"Separated: {len(text_docs)} text documents, {len(image_docs)} image descriptions")
        return text_docs, image_docs
    
    def get_embedding_stats(self, documents: List[Document]) -> dict:
        """
        Get statistics about the documents to be embedded.
        
        Args:
            documents: List of Document objects
        
        Returns:
            Dictionary with statistics
        """
        text_docs, image_docs = self.separate_by_type(documents)
        
        stats = {
            "total_documents": len(documents),
            "text_chunks": len(text_docs),
            "image_descriptions": len(image_docs),
            "authoritative_count": sum(1 for d in documents if d.metadata.get("authoritative", True)),
            "non_authoritative_count": sum(1 for d in documents if not d.metadata.get("authoritative", True)),
            "file_types": {},
            "pages_covered": set()
        }
        
        # Count by file type
        for doc in documents:
            file_type = doc.metadata.get("file_type", "unknown")
            stats["file_types"][file_type] = stats["file_types"].get(file_type, 0) + 1
            
            # Track page coverage
            page_num = doc.metadata.get("page_number")
            if page_num:
                stats["pages_covered"].add((doc.metadata.get("source", ""), page_num))
        
        stats["unique_pages"] = len(stats["pages_covered"])
        del stats["pages_covered"]  # Remove set before returning
        
        return stats


def get_multimodal_embeddings(model_path: str = "nomic-ai/nomic-embed-text-v1.5", device: str = "cpu") -> HuggingFaceEmbeddings:
    """
    Get embeddings for multimodal RAG.
    
    Currently uses the same text embedding model for both text and image descriptions.
    This is the recommended approach for simplicity and unified retrieval.
    
    Args:
        model_path: Path to the embedding model
        device: Device to run on ('cpu' or 'cuda')
    
    Returns:
        HuggingFaceEmbeddings instance
    """
    logger.info(f"Initializing multimodal embeddings with model: {model_path}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': device}
    )
    logger.info("Multimodal embeddings initialized successfully")
    return embeddings


def validate_multimodal_documents(documents: List[Document]) -> bool:
    """
    Validate that multimodal documents have correct structure.
    
    Args:
        documents: List of Document objects
    
    Returns:
        True if all documents are valid
    """
    errors = []
    
    for idx, doc in enumerate(documents):
        # Check for empty content
        if not doc.page_content or not doc.page_content.strip():
            errors.append(f"Document {idx}: Empty content")
        
        # Check metadata
        if not doc.metadata:
            errors.append(f"Document {idx}: Missing metadata")
            continue
        
        # Check image description metadata
        if doc.metadata.get("type") == "image_description":
            required_fields = ["authoritative", "confidence", "page_number", "source"]
            for field in required_fields:
                if field not in doc.metadata:
                    errors.append(f"Document {idx}: Image description missing '{field}'")
            
            # Verify non-authoritative flag
            if doc.metadata.get("authoritative") is not False:
                errors.append(f"Document {idx}: Image description must have authoritative=False")
    
    if errors:
        logger.error(f"Document validation failed with {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10 errors
            logger.error(f"  - {error}")
        return False
    
    logger.info(f"✓ All {len(documents)} documents validated successfully")
    return True


# Example usage and integration notes:
"""
Integration with existing RAG pipeline:

1. Document Parsing (already implemented in document_parser.py):
   - parse_pdf() extracts text chunks + image descriptions
   - parse_docx() extracts text chunks + image descriptions
   - Each image description has type="image_description", authoritative=False

2. Embedding Generation (use this module):
   from services.multimodal_embeddings import get_multimodal_embeddings
   
   embeddings = get_multimodal_embeddings()
   # Same embedding model for both text and images

3. Vector Store Creation:
   from langchain_community.vectorstores import FAISS
   
   vectorstore = FAISS.from_documents(documents, embedding=embeddings)
   # Single index contains both text and image descriptions

4. Retrieval (no changes needed):
   retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
   results = retriever.get_relevant_documents(query)
   # Returns mix of text chunks and image descriptions

5. Post-processing (optional):
   # Filter or rerank based on authoritative flag
   authoritative_results = [d for d in results if d.metadata.get("authoritative", True)]
   supplementary_results = [d for d in results if not d.metadata.get("authoritative", True)]

6. LLM Context Building:
   # Include both authoritative and supplementary content
   # LLM can weight them differently based on metadata
"""


"""
Multi-format document ingestion service for building global FAISS index.
Supports PDF, DOCX, TXT, XLS, XLSX with incremental learning capability.
Includes multimodal support for text and image descriptions.
"""
import os
import logging
from pathlib import Path
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from services.document_parser import parse_document, get_supported_formats
from services.incremental_learning import IncrementalLearningManager, create_or_update_index
from services.multimodal_embeddings import (
    get_multimodal_embeddings,
    validate_multimodal_documents,
    MultimodalEmbeddingManager
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest_single_file(
    file_path: str,
    index_path: str = "snag_faiss_index",
    incremental: bool = True,
    use_ocr: bool = False
) -> dict:
    """
    Ingest a single file into the FAISS index.
    
    Args:
        file_path: Path to the file to ingest
        index_path: Path to FAISS index directory
        incremental: If True, add to existing index; if False, create new
    
    Returns:
        Dictionary with ingestion results
    """
    try:
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"File not found: {file_path}"
            }
        
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()
        
        logger.info(f"📄 Ingesting file: {file_name}" + (" (OCR mode)" if use_ocr else ""))
        
        # Parse document (with OCR if requested)
        documents = parse_document(file_path, use_ocr=use_ocr)
        
        if not documents:
            return {
                "success": False,
                "error": f"No content extracted from {file_name}"
            }
        
        # Validate multimodal documents
        if not validate_multimodal_documents(documents):
            logger.warning("Document validation found issues, but continuing with ingestion")
        
        # Get statistics
        manager_temp = MultimodalEmbeddingManager()
        stats = manager_temp.get_embedding_stats(documents)
        logger.info(f"✓ Parsed {stats['total_documents']} chunks: {stats['text_chunks']} text, {stats['image_descriptions']} images")
        
        # Get multimodal embeddings model
        embeddings = get_multimodal_embeddings(model_path="./all-MiniLM-L6-v2", device='cpu')
        
        # Create or update index
        success = create_or_update_index(
            documents=documents,
            index_path=index_path,
            embeddings=embeddings,
            incremental=incremental
        )
        
        if success:
            logger.info(f"✓ Successfully ingested {file_name}")
            return {
                "success": True,
                "file_name": file_name,
                "file_type": file_ext,
                "num_chunks": len(documents),
                "index_path": index_path,
                "incremental": incremental
            }
        else:
            return {
                "success": False,
                "error": f"Failed to add {file_name} to index"
            }
            
    except Exception as e:
        logger.error(f"✗ Error ingesting {file_path}: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def ingest_directory(
    directory_path: str,
    index_path: str = "snag_faiss_index",
    recursive: bool = True,
    file_extensions: Optional[List[str]] = None
) -> dict:
    """
    Ingest all supported files from a directory into FAISS index.
    
    Args:
        directory_path: Path to directory containing files
        index_path: Path to FAISS index directory
        recursive: If True, search subdirectories
        file_extensions: List of extensions to process (None = all supported)
    
    Returns:
        Dictionary with ingestion results
    """
    try:
        if not os.path.exists(directory_path):
            return {
                "success": False,
                "error": f"Directory not found: {directory_path}"
            }
        
        # Get supported formats
        if file_extensions is None:
            file_extensions = get_supported_formats()
        
        logger.info(f"📁 Scanning directory: {directory_path}")
        logger.info(f"📋 Looking for files: {', '.join(file_extensions)}")
        
        # Find all files
        files_to_process = []
        if recursive:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in file_extensions):
                        files_to_process.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file)
                if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in file_extensions):
                    files_to_process.append(file_path)
        
        if not files_to_process:
            return {
                "success": False,
                "error": "No supported files found in directory"
            }
        
        logger.info(f"📊 Found {len(files_to_process)} files to process")
        
        # Process files
        results = {
            "success": True,
            "total_files": len(files_to_process),
            "processed": 0,
            "failed": 0,
            "files": []
        }
        
        for idx, file_path in enumerate(files_to_process, 1):
            logger.info(f"Processing {idx}/{len(files_to_process)}: {os.path.basename(file_path)}")
            
            result = ingest_single_file(
                file_path=file_path,
                index_path=index_path,
                incremental=True  # Always incremental for batch processing
            )
            
            if result["success"]:
                results["processed"] += 1
            else:
                results["failed"] += 1
            
            results["files"].append(result)
        
        logger.info(f"✓ Batch ingestion complete: {results['processed']} succeeded, {results['failed']} failed")
        
        return results
        
    except Exception as e:
        logger.error(f"✗ Error in batch ingestion: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def rebuild_index_from_scratch(
    source_paths: List[str],
    index_path: str = "snag_faiss_index"
) -> dict:
    """
    Rebuild the entire FAISS index from scratch.
    
    Args:
        source_paths: List of file or directory paths to ingest
        index_path: Path to FAISS index directory
    
    Returns:
        Dictionary with rebuild results
    """
    try:
        logger.info("🔄 Rebuilding FAISS index from scratch...")
        
        # Collect all documents
        all_documents = []
        processed_files = []
        failed_files = []
        
        for source_path in source_paths:
            if os.path.isfile(source_path):
                # Single file
                try:
                    docs = parse_document(source_path)
                    all_documents.extend(docs)
                    processed_files.append(source_path)
                    logger.info(f"✓ Processed: {os.path.basename(source_path)} ({len(docs)} chunks)")
                except Exception as e:
                    failed_files.append({"path": source_path, "error": str(e)})
                    logger.error(f"✗ Failed: {os.path.basename(source_path)} - {str(e)}")
            
            elif os.path.isdir(source_path):
                # Directory
                supported_formats = get_supported_formats()
                for root, dirs, files in os.walk(source_path):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in supported_formats):
                            file_path = os.path.join(root, file)
                            try:
                                docs = parse_document(file_path)
                                all_documents.extend(docs)
                                processed_files.append(file_path)
                                logger.info(f"✓ Processed: {os.path.basename(file_path)} ({len(docs)} chunks)")
                            except Exception as e:
                                failed_files.append({"path": file_path, "error": str(e)})
                                logger.error(f"✗ Failed: {os.path.basename(file_path)} - {str(e)}")
        
        if not all_documents:
            return {
                "success": False,
                "error": "No documents were successfully processed"
            }
        
        # Validate multimodal documents
        if not validate_multimodal_documents(all_documents):
            logger.warning("Some documents have validation issues, but continuing with index building")
        
        # Get statistics
        manager_temp = MultimodalEmbeddingManager()
        stats = manager_temp.get_embedding_stats(all_documents)
        logger.info(f"📊 Total documents: {stats['total_documents']} ({stats['text_chunks']} text, {stats['image_descriptions']} images)")
        
        # Create multimodal embeddings
        embeddings = get_multimodal_embeddings(model_path="./all-MiniLM-L6-v2", device='cpu')
        
        # Create new index
        logger.info("🔨 Building FAISS index...")
        vectorstore = FAISS.from_documents(all_documents, embedding=embeddings)
        
        # Save index
        vectorstore.save_local(index_path)
        logger.info(f"✓ Index saved to: {index_path}")
        
        # Initialize metadata
        manager = IncrementalLearningManager(index_path, embeddings)
        manager.metadata["total_documents"] = len(all_documents)
        manager._save_metadata()
        
        return {
            "success": True,
            "total_documents": len(all_documents),
            "processed_files": len(processed_files),
            "failed_files": len(failed_files),
            "index_path": index_path,
            "files": {
                "processed": processed_files,
                "failed": failed_files
            }
        }
        
    except Exception as e:
        logger.error(f"✗ Error rebuilding index: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def get_index_info(index_path: str = "snag_faiss_index") -> dict:
    """
    Get information about the current FAISS index.
    
    Args:
        index_path: Path to FAISS index directory
    
    Returns:
        Dictionary with index information
    """
    try:
        if not os.path.exists(index_path):
            return {
                "exists": False,
                "error": "Index does not exist"
            }
        
        model_path = "./all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cpu'}
        )
        
        manager = IncrementalLearningManager(index_path, embeddings)
        stats = manager.get_statistics()
        sources = manager.list_sources()
        
        return {
            "exists": True,
            "statistics": stats,
            "sources": sources,
            "num_sources": len(sources)
        }
        
    except Exception as e:
        logger.error(f"Error getting index info: {str(e)}")
        return {
            "exists": False,
            "error": str(e)
        }


# Legacy function for backward compatibility
def ingest():
    """Legacy function - ingests default data file"""
    default_file = "data/confidential_snag.xlsx"
    if os.path.exists(default_file):
        result = ingest_single_file(default_file, incremental=False)
        if result["success"]:
            logger.info("✓ Default data ingested successfully")
        else:
            logger.error(f"✗ Failed to ingest default data: {result.get('error')}")
    else:
        logger.warning(f"Default data file not found: {default_file}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line usage
        command = sys.argv[1]
        
        if command == "file" and len(sys.argv) > 2:
            # Ingest single file
            file_path = sys.argv[2]
            incremental = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else True
            result = ingest_single_file(file_path, incremental=incremental)
            print(result)
        
        elif command == "dir" and len(sys.argv) > 2:
            # Ingest directory
            dir_path = sys.argv[2]
            result = ingest_directory(dir_path)
            print(result)
        
        elif command == "rebuild" and len(sys.argv) > 2:
            # Rebuild index
            paths = sys.argv[2:]
            result = rebuild_index_from_scratch(paths)
            print(result)
        
        elif command == "info":
            # Get index info
            result = get_index_info()
            print(result)
        
        else:
            print("Usage:")
            print("  python ingest.py file <path> [incremental]")
            print("  python ingest.py dir <path>")
            print("  python ingest.py rebuild <path1> [path2] ...")
            print("  python ingest.py info")
    else:
        # Default behavior
        ingest()


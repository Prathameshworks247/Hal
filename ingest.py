# app.py
import os
import logging
from typing import Dict, List, Tuple, Any
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from llm import get_llm

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_chain():
    try:
        logger.info("Initializing embeddings model...")
        # Check if the model path exists
        model_path = "./all-MiniLM-L6-v2"
        if not os.path.exists(model_path):
            logger.warning(f"Local model path {model_path} not found. Using HuggingFace model name instead.")
            model_path = "sentence-transformers/all-MiniLM-L6-v2"
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cpu'}  # Explicitly set device
        )
        logger.info("Embeddings model loaded successfully.")

        logger.info("Loading FAISS index...")
        # Check if FAISS index exists
        faiss_index_path = "snag_faiss_index"
        if not os.path.exists(faiss_index_path):
            raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")
        
        db = FAISS.load_local(
            faiss_index_path, 
            embeddings=embeddings, 
            allow_dangerous_deserialization=True
        )
        logger.info("FAISS index loaded successfully.")

        # Test the retriever
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})  # Increased to 5 for more context
        logger.info("Retriever configured successfully.")

        # Enhanced prompt template
        prompt = PromptTemplate.from_template("""
        You are an expert aircraft technician with extensive experience in aircraft maintenance and troubleshooting.
        
        Based on the following historical snag records and their rectifications, provide a detailed recommendation for fixing the current snag.
        
        Current Snag: {question}
        
        Historical Snag Records:
        {context}
        
        Please provide:
        1. Most likely cause of the issue
        2. Step-by-step rectification procedure
        3. Any safety precautions to consider
        4. Parts that might need replacement
        5. Expected time to complete the fix
        
        Recommended Rectification:
        """)

        logger.info("Getting LLM instance...")
        llm = get_llm()
        logger.info("LLM instance obtained successfully.")

        # Create the QA chain with proper input/output keys
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": prompt,
                "verbose": True  # Enable verbose mode for debugging
            },
            return_source_documents=True,  # Return source documents for transparency
            input_key="question",  # Explicitly set input key
            output_key="result"    # Explicitly set output key
        )
        
        logger.info("QA chain created successfully.")
        return qa_chain, db

    except Exception as e:
        logger.error(f"Error in get_chain(): {str(e)}")
        raise

def get_similar_snags_with_metadata(db, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve similar snags with their metadata and similarity scores
    
    Args:
        db: FAISS database instance
        query: Current snag description
        k: Number of similar documents to retrieve
        
    Returns:
        List of dictionaries containing document content, metadata, and similarity scores
    """
    try:
        # Get retriever
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
        
        # Get similar documents
        similar_docs = retriever.get_relevant_documents(query)
        
        # Also get similarity scores using similarity_search_with_score
        docs_with_scores = db.similarity_search_with_score(query, k=k)
        
        # Combine documents with scores and metadata
        similar_snags = []
        for i, (doc, score) in enumerate(docs_with_scores):
            snag_info = {
                'rank': i + 1,
                'content': doc.page_content,
                'metadata': doc.metadata if hasattr(doc, 'metadata') else {},
                'similarity_score': float(score),
                'similarity_percentage': round((1 - score) * 100, 2)  # Convert distance to similarity percentage
            }
            similar_snags.append(snag_info)
        
        return similar_snags
        
    except Exception as e:
        logger.error(f"Error retrieving similar snags: {str(e)}")
        return []

def process_snag_query(chain, db, query: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Process snag query and return both rectification and similar snags with metadata
    
    Args:
        chain: QA chain instance
        db: FAISS database instance
        query: Snag description
        
    Returns:
        Tuple of (rectification_response, similar_snags_list)
    """
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
        
        return rectification, similar_snags
        
    except Exception as e:
        logger.error(f"Error processing snag query: {str(e)}")
        return f"Error: {str(e)}", []

def display_similar_snags(similar_snags: List[Dict[str, Any]]):
    """Display similar snags with their metadata in a formatted way"""
    if not similar_snags:
        print("❌ No similar snags found.")
        return
    
    print(f"\n📊 Found {len(similar_snags)} Similar Previous Snags:")
    print("=" * 80)
    
    for snag in similar_snags:
        print(f"\n🔍 Rank #{snag['rank']} (Similarity: {snag['similarity_percentage']}%)")
        print("-" * 60)
        
        # Display content (truncated)
        content = snag['content']
        if len(content) > 200:
            content = content[:200] + "..."
        print(f"📝 Content: {content}")
        
        # Display metadata
        metadata = snag['metadata']
        if metadata:
            print("📋 Metadata:")
            for key, value in metadata.items():
                print(f"   • {key}: {value}")
        else:
            print("📋 Metadata: None available")
        
        print(f"📈 Similarity Score: {snag['similarity_score']:.4f}")
        print()

def test_retriever(db, query):
    """Test function to check if retriever is working"""
    try:
        print(f"\n🧪 Testing retriever with query: '{query}'")
        similar_snags = get_similar_snags_with_metadata(db, query, k=3)
        
        if similar_snags:
            print(f"✅ Retriever test successful! Found {len(similar_snags)} similar documents.")
            display_similar_snags(similar_snags[:2])  # Show top 2 for testing
            return True
        else:
            print("❌ No documents retrieved.")
            return False
            
    except Exception as e:
        print(f"❌ Retriever test failed: {e}")
        return False

def export_results_to_dict(rectification: str, similar_snags: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Export results in a structured dictionary format for API usage or further processing
    """
    return {
        'rectification': {
            'recommendation': rectification,
            'generated_at': str(logging.time.time())
        },
        'similar_snags': similar_snags,
        'summary': {
            'total_similar_snags': len(similar_snags),
            'avg_similarity': round(sum(s['similarity_percentage'] for s in similar_snags) / len(similar_snags), 2) if similar_snags else 0,
            'most_similar_score': similar_snags[0]['similarity_percentage'] if similar_snags else 0
        }
    }

def main():
    try:
        print("🚁 Aircraft Snag Resolution System with Metadata")
        print("=" * 50)
        
        # Initialize the chain and database
        print("🔧 Initializing system...")
        chain, db = get_chain()
        print("✅ System initialized successfully!")
        
        # Test the system
        test_query = "hydraulic system pressure low"
        print(f"\n🧪 Testing system with query: '{test_query}'")
        test_retriever(db, test_query)
        
        while True:
            try:
                query = input("\n🛠️ Enter new snag (or type 'exit' to quit): ").strip()
                
                if query.lower() in ['exit', 'quit', 'q']:
                    print("🔚 Exiting system. Goodbye!")
                    break
                
                if not query:
                    print("⚠️ Please enter a valid snag description.")
                    continue

                print("\n🔍 Analyzing snag and fetching recommendations...")
                
                # Process the query
                rectification, similar_snags = process_snag_query(chain, db, query)
                
                # Display results
                print("\n✅ AI-Generated Rectification:")
                print("=" * 60)
                print(rectification)
                
                # Display similar snags with metadata
                display_similar_snags(similar_snags)
                
                # Option to export results
                export_choice = input("\n💾 Export results to dictionary format? (y/n): ").strip().lower()
                if export_choice in ['y', 'yes']:
                    results_dict = export_results_to_dict(rectification, similar_snags)
                    print("\n📄 Exported Results:")
                    print("-" * 40)
                    print(f"Rectification Length: {len(results_dict['rectification']['recommendation'])} characters")
                    print(f"Similar Snags Found: {results_dict['summary']['total_similar_snags']}")
                    print(f"Average Similarity: {results_dict['summary']['avg_similarity']}%")
                    print(f"Most Similar Score: {results_dict['summary']['most_similar_score']}%")
                    
                    # Optionally save to file
                    save_choice = input("💾 Save to JSON file? (y/n): ").strip().lower()
                    if save_choice in ['y', 'yes']:
                        import json
                        from datetime import datetime
                        filename = f"snag_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        with open(filename, 'w') as f:
                            json.dump(results_dict, f, indent=2)
                        print(f"✅ Results saved to {filename}")
                        
            except KeyboardInterrupt:
                print("\n🔚 Interrupted by user. Exiting.")
                break
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                print(f"\n❌ Error processing your request: {e}")
                print("Please try rephrasing your query or check the system logs.")

    except Exception as e:
        logger.error(f"Fatal error in main(): {str(e)}")
        print(f"\n💥 System initialization failed: {e}")
        print("\nTroubleshooting steps:")
        print("1. Check if 'snag_faiss_index' directory exists")
        print("2. Verify the embedding model is available")
        print("3. Ensure the LLM is properly configured in llm.py")
        print("4. Check file permissions")

if __name__ == "__main__":
    main()
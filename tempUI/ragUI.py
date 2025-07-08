# streamlit_app.py
import streamlit as st
import os
import logging
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from services.llm import get_llm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Aircraft Snag Resolution System",
    page_icon="🛩️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .snag-input {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .result-container {
        background-color: #e8f5e8;
        border-left: 4px solid #28a745;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .error-container {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .info-container {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .source-doc {
        background-color: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_chain():
    """Initialize the RAG chain with caching for better performance"""
    try:
        with st.spinner("🔧 Initializing AI system..."):
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Loading embedding model...")
            progress_bar.progress(25)
            
            # Check if the model path exists
            model_path = "./all-MiniLM-L6-v2"
            if not os.path.exists(model_path):
                logger.warning(f"Local model path {model_path} not found. Using HuggingFace model.")
                model_path = "sentence-transformers/all-MiniLM-L6-v2"
            
            embeddings = HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={'device': 'cpu'}
            )
            
            status_text.text("Loading knowledge base...")
            progress_bar.progress(50)
            
            # Check if FAISS index exists
            faiss_index_path = "snag_faiss_index"
            if not os.path.exists(faiss_index_path):
                raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")
            
            db = FAISS.load_local(
                faiss_index_path, 
                embeddings=embeddings, 
                allow_dangerous_deserialization=True
            )
            
            status_text.text("Configuring retrieval system...")
            progress_bar.progress(75)
            
            retriever = db.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": st.session_state.get('num_sources', 3)}
            )
            
            # Enhanced prompt template
            prompt = PromptTemplate.from_template("""
            You are an expert aircraft technician with extensive experience in aircraft maintenance and troubleshooting.
            
            Based on the following historical snag records and their rectifications, provide a detailed recommendation for fixing the current snag.
            
            Current Snag: {question}
            
            Historical Snag Records:
            {context}
            
            Please provide:
            1. **Most likely cause** of the issue
            2. **Step-by-step rectification procedure**
            3. **Safety precautions** to consider
            4. **Parts** that might need replacement
            5. **Expected time** to complete the fix
            6. **Additional recommendations** or preventive measures
            
            Format your response clearly with proper headings and bullet points for easy reading.
            
            Recommended Rectification:
            """)
            
            status_text.text("Initializing language model...")
            progress_bar.progress(90)
            
            llm = get_llm()
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True,
                input_key="question",
                output_key="result"
            )
            
            progress_bar.progress(100)
            status_text.text("✅ System ready!")
            
            # Clear progress indicators after a short delay
            import time
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            return qa_chain, db
            
    except Exception as e:
        logger.error(f"Error initializing chain: {str(e)}")
        st.error(f"Failed to initialize system: {str(e)}")
        return None, None

def display_header():
    """Display the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🛩️ Aircraft Snag Resolution System</h1>
        <p>AI-Powered Aircraft Maintenance Support</p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display sidebar with configuration options"""
    st.sidebar.header("⚙️ Configuration")
    
    # Number of similar cases to retrieve
    num_sources = st.sidebar.slider(
        "Number of historical cases to analyze",
        min_value=1,
        max_value=10,
        value=3,
        help="More cases provide broader context but may slow response time"
    )
    st.session_state['num_sources'] = num_sources
    
    # Display source documents toggle
    show_sources = st.sidebar.checkbox(
        "Show source documents",
        value=True,
        help="Display the historical cases used for recommendations"
    )
    st.session_state['show_sources'] = show_sources
    
    # Advanced options
    with st.sidebar.expander("🔧 Advanced Options"):
        st.session_state['detailed_response'] = st.checkbox(
            "Request detailed analysis",
            value=True,
            help="Get more comprehensive rectification procedures"
        )
        
        st.session_state['include_safety'] = st.checkbox(
            "Emphasize safety procedures",
            value=True,
            help="Include additional safety considerations"
        )
    
    # System information
    st.sidebar.header("📊 System Info")
    if 'chain' in st.session_state and st.session_state.chain:
        st.sidebar.success("✅ System Online")
        if 'query_count' in st.session_state:
            st.sidebar.info(f"Queries processed: {st.session_state.query_count}")
    else:
        st.sidebar.error("❌ System Offline")
    
    # Help section
    with st.sidebar.expander("❓ Help & Examples"):
        st.markdown("""
        **Example Snag Descriptions:**
        - "Hydraulic system pressure low"
        - "Engine oil temperature high during climb"
        - "Landing gear retraction warning light on"
        - "Radio communication intermittent"
        - "Fuel pump pressure fluctuating"
        
        **Tips for better results:**
        - Be specific about the aircraft system
        - Include symptoms and conditions
        - Mention any error codes or warnings
        """)

def process_query(chain, query):
    """Process the snag query and return results"""
    try:
        with st.spinner("🔍 Analyzing snag and searching historical records..."):
            response = chain.invoke({"question": query})
            
            if isinstance(response, dict):
                result = response.get('result', response.get('answer', str(response)))
                sources = response.get('source_documents', [])
                return result, sources
            else:
                return str(response), []
                
    except AttributeError:
        # Fallback for older LangChain versions
        try:
            with st.spinner("🔍 Processing your query..."):
                result = chain.run(query)
                return result, []
        except Exception as e:
            raise e
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise e

def display_results(result, sources):
    """Display the analysis results"""
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    st.markdown("## ✅ Recommended Rectification")
    st.markdown(result)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display source documents if enabled
    if st.session_state.get('show_sources', True) and sources:
        st.markdown("---")
        st.markdown("## 📚 Historical Cases Referenced")
        
        for i, doc in enumerate(sources):
            with st.expander(f"📄 Historical Case {i+1}", expanded=False):
                st.markdown(f'<div class="source-doc">{doc.page_content}</div>', 
                           unsafe_allow_html=True)
                if hasattr(doc, 'metadata') and doc.metadata:
                    st.markdown("**Metadata:**")
                    st.json(doc.metadata)

def main():
    """Main application function"""
    display_header()
    display_sidebar()
    
    # Initialize session state
    if 'query_count' not in st.session_state:
        st.session_state.query_count = 0
    
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    # Initialize the system
    if 'chain' not in st.session_state or st.session_state.chain is None:
        st.session_state.chain, st.session_state.db = initialize_chain()
    
    if st.session_state.chain is None:
        st.markdown('<div class="error-container">', unsafe_allow_html=True)
        st.error("❌ System initialization failed. Please check the following:")
        st.markdown("""
        1. Ensure `snag_faiss_index` directory exists
        2. Verify the embedding model is available
        3. Check that `llm.py` is properly configured
        4. Ensure all dependencies are installed
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="snag-input">', unsafe_allow_html=True)
        st.markdown("### 🛠️ Describe the Aircraft Snag")
        
        # Text input for snag description
        query = st.text_area(
            "Enter detailed snag description:",
            height=100,
            placeholder="Example: Hydraulic system pressure reading 2000 PSI, normal is 3000 PSI. Warning light illuminated during preflight check.",
            help="Provide as much detail as possible for better recommendations"
        )
        
        # Quick example buttons
        st.markdown("**Quick Examples:**")
        col_ex1, col_ex2, col_ex3 = st.columns(3)
        
        with col_ex1:
            if st.button("🔧 Hydraulic Issue"):
                query = "Hydraulic system pressure low, warning light on"
                
        with col_ex2:
            if st.button("⚡ Electrical Problem"):
                query = "Navigation lights intermittent, flickering during flight"
                
        with col_ex3:
            if st.button("🛞 Landing Gear Issue"):
                query = "Landing gear retraction warning, gear position indicator unclear"
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process button
        if st.button("🚀 Get Rectification Recommendation", type="primary", use_container_width=True):
            if query.strip():
                try:
                    result, sources = process_query(st.session_state.chain, query)
                    
                    # Update session state
                    st.session_state.query_count += 1
                    st.session_state.query_history.append({
                        'timestamp': datetime.now(),
                        'query': query,
                        'result': result[:200] + "..." if len(result) > 200 else result
                    })
                    
                    # Display results
                    display_results(result, sources)
                    
                except Exception as e:
                    st.markdown('<div class="error-container">', unsafe_allow_html=True)
                    st.error(f"❌ Error processing your request: {str(e)}")
                    st.markdown("Please try rephrasing your query or contact system administrator.")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please enter a snag description before submitting.")
    
    with col2:
        st.markdown("### 📈 Quick Stats")
        
        # System status card
        st.markdown('<div class="info-container">', unsafe_allow_html=True)
        st.metric("System Status", "Online ✅")
        st.metric("Queries Processed", st.session_state.query_count)
        st.metric("Knowledge Base", "Ready 📚")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Recent queries
        if st.session_state.query_history:
            st.markdown("### 🕐 Recent Queries")
            for i, item in enumerate(reversed(st.session_state.query_history[-3:])):
                with st.expander(f"Query {len(st.session_state.query_history)-i}"):
                    st.markdown(f"**Time:** {item['timestamp'].strftime('%H:%M:%S')}")
                    st.markdown(f"**Query:** {item['query'][:100]}...")
                    st.markdown(f"**Result:** {item['result']}")

if __name__ == "__main__":
    main()
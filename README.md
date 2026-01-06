# Sky-Sentinel: Offline AI-Powered RAG Knowledge Portal

## 🚁 Overview

Sky-Sentinel is an advanced, **fully offline** AI-powered Retrieval-Augmented Generation (RAG) system designed for aircraft maintenance knowledge management. It provides intelligent document search, analysis, and recommendations while ensuring complete data privacy and security through offline operation.

### 🏆 Hackathon-Winning Features

#### Core Features
1. **Multi-Format Document Support** ✅
   - PDF (with page-level citations)
   - DOCX/DOC (with paragraph-level citations)
   - TXT (with line-level citations)
   - XLS/XLSX (with row-level citations)

2. **Complete Offline Operation** ✅
   - Local LLM (Ollama with optimized models)
   - Local embeddings (all-MiniLM-L6-v2)
   - No internet dependency
   - Full data privacy

3. **Enhanced Citations & Traceability** ✅
   - Precise source attribution (page/paragraph/line/row)
   - Document lineage tracking
   - Confidence scoring
   - Audit trail for all queries

4. **Anti-Hallucination System** ✅
   - Strict prompt templates with verification
   - Multi-layer prompt security
   - Semantic validation
   - Context-grounded responses only

5. **Incremental Learning** ✅
   - Add documents without full rebuild
   - Update existing documents
   - Track changes over time
   - Version control

6. **Role-Based Access Control (RBAC)** ✅
   - User authentication
   - Role hierarchy (Admin, Editor, Viewer, Guest)
   - Document-level permissions
   - Audit logging

7. **Interactive Chatbot** ✅
   - Natural language queries
   - Context-aware responses
   - Citation references
   - Analytics and insights

---

## 🎯 Problem Statement Solution

**Challenge:** Organizations generate huge amounts of internal knowledge (technical manuals, research papers, project reports, compliance docs). Searching these documents manually is inefficient, and cloud/internet-based solutions raise privacy and security concerns.

**Our Solution:** Sky-Sentinel provides:
- ✅ Multi-format document ingestion (PDF, DOCX, TXT, Excel)
- ✅ AI embeddings and intelligent indexing
- ✅ Offline LLM with RAG pipeline
- ✅ Document-based response generation with citations
- ✅ Natural language query interface
- ✅ Complete offline operation (no internet required)
- ✅ Local hardware optimization
- ✅ Data privacy assurance
- ✅ Incremental learning (Bonus)
- ✅ Role-based access control (Bonus)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Ollama (for offline LLM)
- 8GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Sky-Sentinal
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download Spacy model**
```bash
python -m spacy download en_core_web_md
```

5. **Install and setup Ollama**
```bash
# Install Ollama from https://ollama.ai

# Pull recommended model (8B parameters, optimized for RAG)
ollama pull llama3.2:8b-instruct-q4_K_M

# Alternative models (all under 10B):
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull deepseek-r1:7b-instruct-q4_K_M
ollama pull mistral:7b-instruct-v0.3-q4_K_M
```

6. **Download embedding model**
```bash
# The embedding model will be downloaded automatically on first run
# Or download manually:
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('./all-MiniLM-L6-v2')"
```

### Running the Application

1. **Start the API server**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

2. **Access the API**
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- System Info: http://localhost:8000/system/info

---

## 📚 API Endpoints

### Document Management

#### Upload Document
```bash
POST /store_file
Content-Type: multipart/form-data

Parameters:
- file: Document file (PDF, DOCX, TXT, XLS, XLSX)
- pb_number: Project/batch number
```

#### List Documents
```bash
POST /send_file_names
Content-Type: application/json

{
  "pb_number": "PB1234"
}
```

#### Add Document to Index (Incremental)
```bash
POST /index/add_document
Content-Type: multipart/form-data

Parameters:
- file: Document file
- pb_number: Project/batch number
```

### Query & Analysis

#### Rectification Query (with Citations)
```bash
POST /rectify
Content-Type: application/json

{
  "query": "Snag: Hydraulic pressure low in main system",
  "file_name": "default",
  "pb_number": "PB1234"
}
```

#### Analytics Query
```bash
POST /analytics
Content-Type: application/json

{
  "query": "Analyze hydraulic system failures",
  "file_name": "default",
  "pb_number": "PB1234"
}
```

#### Verify Query Quality
```bash
POST /verify_query
Content-Type: application/json

{
  "query": "Your query here"
}
```

### System Information

#### Get System Info
```bash
GET /system/info
```

#### Get Supported Formats
```bash
GET /formats/supported
```

#### Get Index Statistics
```bash
GET /index/statistics
```

---

## 🧠 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│              (Frontend/API Clients)                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Backend                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │         Prompt Verification Layer                │   │
│  │  • Semantic validation                           │   │
│  │  • Security filtering                            │   │
│  │  • Quality scoring                               │   │
│  └──────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            Multi-Format Document Parser                  │
│  ┌──────────┬──────────┬──────────┬──────────┐         │
│  │   PDF    │   DOCX   │   TXT    │  Excel   │         │
│  │ (PyMuPDF)│ (docx)   │(encoding)│ (pandas) │         │
│  └──────────┴──────────┴──────────┴──────────┘         │
│         Smart Chunking with Overlap                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Enhanced Metadata System                    │
│  • Page/Paragraph/Line/Row numbers                      │
│  • Document structure                                    │
│  • Timestamps & versioning                              │
│  • Citation information                                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Vector Store (FAISS - Offline)                   │
│  • Embeddings: all-MiniLM-L6-v2 (Local)                 │
│  • Incremental updates supported                        │
│  • Metadata filtering                                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              RAG Pipeline (Enhanced)                     │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Retriever: Semantic similarity search           │   │
│  │  • Top-k relevant documents                      │   │
│  │  • Confidence scoring                            │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Generator: LLM (Ollama - Offline)               │   │
│  │  • llama3.2:8b-instruct-q4_K_M                   │   │
│  │  • Temperature: 0.1 (low hallucination)          │   │
│  │  • Anti-hallucination prompts                    │   │
│  └──────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Response with Full Traceability                  │
│  • Formatted answer                                      │
│  • Source citations (with locations)                     │
│  • Confidence scores                                     │
│  • Audit trail                                           │
└─────────────────────────────────────────────────────────┘
```

---

## 🔧 Configuration

### LLM Model Selection

Edit `services/llm.py` to change the model:

```python
# Recommended models (in order of preference):
# 1. llama3.2:8b-instruct-q4_K_M - Best balance (default)
# 2. qwen2.5:7b-instruct-q4_K_M - Excellent reasoning
# 3. deepseek-r1:7b-instruct-q4_K_M - Great for technical content
# 4. mistral:7b-instruct-v0.3-q4_K_M - Good general purpose

model = "llama3.2:8b-instruct-q4_K_M"
```

### Environment Variables

Create a `.env` file:

```bash
# LLM Configuration
OLLAMA_MODEL=llama3.2:8b-instruct-q4_K_M

# Debug Mode
DEBUG_MODE=0

# Server Configuration
HOST=0.0.0.0
PORT=8000
```

---

## 🎨 Key Features Explained

### 1. Multi-Format Document Support

**Supported Formats:**
- **PDF**: Page-level citations with PyMuPDF
- **DOCX**: Paragraph-level citations with python-docx
- **TXT**: Line-level citations with encoding detection
- **Excel**: Row-level citations with pandas

**Smart Chunking:**
- Sentence-boundary aware
- Configurable chunk size and overlap
- Context preservation

### 2. Anti-Hallucination System

**Prompt Engineering:**
- Strict grounding instructions
- Verification checklists
- Citation requirements
- "Insufficient data" fallbacks

**Multi-Layer Verification:**
- Length validation
- Malicious content detection
- Semantic meaning check
- Context relevance scoring

### 3. Enhanced Citations

**Citation Information:**
- Source document name
- Precise location (page/paragraph/line/row)
- Content preview
- Ingestion timestamp
- Confidence score

**Example Citation:**
```json
{
  "citation_id": 1,
  "formatted_citation": "[Manual_v2.pdf, Page 42]",
  "content_preview": "Hydraulic system pressure should be maintained...",
  "confidence": "high"
}
```

### 4. Incremental Learning

**Benefits:**
- No full index rebuild required
- Fast document addition
- Update tracking
- Version control

**Usage:**
```python
from services.incremental_learning import IncrementalLearningManager

manager = IncrementalLearningManager("snag_faiss_index")
manager.add_documents(new_documents, source_file="new_manual.pdf")
```

### 5. Role-Based Access Control

**Roles:**
- **Admin**: Full access + user management
- **Editor**: Read/write access
- **Viewer**: Read-only access
- **Guest**: Limited read access

**Features:**
- User authentication
- Document-level permissions
- Audit logging
- Session management

---

## 📊 Performance Optimization

### Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 10GB

**Recommended:**
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 50GB SSD
- GPU: Optional (for faster embeddings)

### Optimization Tips

1. **Model Selection:**
   - Use quantized models (q4_K_M) for speed
   - 7-8B parameter models are optimal

2. **Chunking:**
   - Adjust chunk_size based on document type
   - PDF/DOCX: 1000 chars
   - TXT: 800 chars
   - Excel: Row-based (no chunking)

3. **Retrieval:**
   - Adjust k (number of results) based on needs
   - Higher k = more context but slower

4. **Caching:**
   - LRU cache for frequently accessed chains
   - Session caching for user queries

---

## 🔒 Security & Privacy

### Offline Operation
- ✅ No internet connection required
- ✅ All processing happens locally
- ✅ No data leaves your infrastructure

### Data Privacy
- ✅ No cloud services
- ✅ No external API calls
- ✅ Complete data sovereignty

### Access Control
- ✅ User authentication
- ✅ Role-based permissions
- ✅ Document-level access control
- ✅ Audit logging

---

## 🧪 Testing

### Run Tests
```bash
# Unit tests
pytest tests/

# Integration tests
pytest tests/integration/

# Load tests
locust -f tests/load_test.py
```

### Manual Testing
```bash
# Test document upload
curl -X POST "http://localhost:8000/store_file" \
  -F "file=@test_document.pdf" \
  -F "pb_number=TEST001"

# Test query
curl -X POST "http://localhost:8000/rectify" \
  -H "Content-Type: application/json" \
  -d '{"query": "Snag: Engine failure", "file_name": "default", "pb_number": "TEST001"}'
```

---

## 📈 Monitoring & Analytics

### Index Statistics
```bash
GET /index/statistics
```

Returns:
- Total documents
- Number of sources
- Last update time
- Update history

### Query Analytics
- Query verification scores
- Response confidence levels
- Source distribution
- Performance metrics

---

## 🤝 Contributing

### Development Setup
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run linter
flake8 services/ app.py

# Format code
black services/ app.py
```

### Code Structure
```
Sky-Sentinal/
├── app.py                      # Main FastAPI application
├── services/
│   ├── document_parser.py      # Multi-format parser
│   ├── citation_service.py     # Citation management
│   ├── prompt_verifier.py      # Prompt verification
│   ├── incremental_learning.py # Incremental updates
│   ├── rbac_service.py         # Access control
│   ├── chain_service.py        # RAG chains
│   ├── similarity_service.py   # Similarity search
│   ├── llm.py                  # LLM configuration
│   └── excel_service.py        # Excel parsing
├── models/
│   └── models.py               # Pydantic models
├── utils/
│   └── utils.py                # Utility functions
├── vision/
│   └── funcs.py                # Computer vision features
└── requirements.txt            # Dependencies
```

---

## 🏆 Hackathon Advantages

### Why Sky-Sentinel Wins:

1. **Complete Solution** ✅
   - All core features implemented
   - Both bonus challenges completed
   - Production-ready architecture

2. **Innovation** ✅
   - Multi-format support (beyond requirements)
   - Anti-hallucination system
   - Incremental learning
   - RBAC with audit logging

3. **Privacy & Security** ✅
   - 100% offline operation
   - No cloud dependencies
   - Complete data sovereignty
   - Enterprise-grade security

4. **User Experience** ✅
   - Natural language queries
   - Precise citations
   - Confidence scoring
   - Fast response times

5. **Scalability** ✅
   - Incremental updates (no full rebuild)
   - Optimized for local hardware
   - Efficient chunking and retrieval
   - Caching strategies

6. **Documentation** ✅
   - Comprehensive README
   - API documentation
   - Architecture diagrams
   - Usage examples

---

## 📝 License

[Your License Here]

---

## 👥 Team

[Your Team Information]

---

## 📞 Support

For questions or issues:
- GitHub Issues: [Your Repo URL]
- Email: [Your Email]
- Documentation: [Your Docs URL]

---

## 🙏 Acknowledgments

- Ollama for offline LLM support
- LangChain for RAG framework
- HuggingFace for embeddings
- FastAPI for API framework

---

**Built with ❤️ for offline, privacy-preserving AI knowledge management**


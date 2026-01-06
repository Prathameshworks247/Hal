# Installation Guide - Sky-Sentinel

## Step-by-Step Installation

### 1. System Requirements

**Minimum Requirements:**
- Operating System: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- Python: 3.10 or higher
- RAM: 8GB minimum (16GB recommended)
- Storage: 10GB free space (50GB recommended)
- CPU: 4 cores minimum (8+ cores recommended)

**Optional:**
- GPU: NVIDIA GPU with CUDA support (for faster embeddings)

### 2. Install Python

#### Windows:
```bash
# Download from python.org or use winget
winget install Python.Python.3.11
```

#### macOS:
```bash
# Using Homebrew
brew install python@3.11
```

#### Linux:
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

### 3. Clone Repository

```bash
git clone <your-repository-url>
cd Sky-Sentinal
```

### 4. Create Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 5. Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

**Note:** If you encounter issues with specific packages:

```bash
# For PDF support (choose one):
pip install pymupdf  # Recommended
# OR
pip install pdfplumber

# For DOCX support:
pip install python-docx

# For computer vision features:
pip install opencv-python
```

### 6. Download Spacy Language Model

```bash
# Download English language model (medium size)
python -m spacy download en_core_web_md

# Verify installation
python -c "import spacy; nlp = spacy.load('en_core_web_md'); print('✓ Spacy model loaded successfully')"
```

### 7. Install and Setup Ollama

#### Windows:
```bash
# Download installer from https://ollama.ai/download/windows
# Run the installer

# Verify installation
ollama --version
```

#### macOS:
```bash
# Download from https://ollama.ai/download/mac
# Or use Homebrew
brew install ollama

# Verify installation
ollama --version
```

#### Linux:
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Verify installation
ollama --version
```

### 8. Download LLM Models

```bash
# Start Ollama service (if not already running)
# On Windows/macOS: Ollama runs automatically
# On Linux:
ollama serve &

# Pull recommended model (8B parameters, ~4.7GB)
ollama pull llama3.2:8b-instruct-q4_K_M

# Verify model
ollama list

# Optional: Pull alternative models
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull mistral:7b-instruct-v0.3-q4_K_M
```

### 9. Download Embedding Model

The embedding model will be downloaded automatically on first run. To pre-download:

```bash
python -c "
from sentence_transformers import SentenceTransformer
print('Downloading embedding model...')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.save('./all-MiniLM-L6-v2')
print('✓ Embedding model downloaded successfully')
"
```

### 10. Create Required Directories

```bash
# Create directories for data storage
mkdir -p uploaded_excels
mkdir -p static
mkdir -p outputs
mkdir -p uploads
mkdir -p rbac_data
mkdir -p snag_faiss_index
```

### 11. (Optional) Create Initial Vector Index

If you have existing data:

```bash
python services/ingest.py
```

### 12. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# LLM Configuration
OLLAMA_MODEL=llama3.2:8b-instruct-q4_K_M

# Server Configuration
HOST=0.0.0.0
PORT=8000

# Debug Mode (0 = off, 1 = on)
DEBUG_MODE=0

# Logging Level
LOG_LEVEL=INFO
```

### 13. Verify Installation

Run the verification script:

```bash
python -c "
import sys
print('Python version:', sys.version)

try:
    import fastapi
    print('✓ FastAPI installed')
except:
    print('✗ FastAPI not installed')

try:
    import langchain
    print('✓ LangChain installed')
except:
    print('✗ LangChain not installed')

try:
    import sentence_transformers
    print('✓ Sentence Transformers installed')
except:
    print('✗ Sentence Transformers not installed')

try:
    import faiss
    print('✓ FAISS installed')
except:
    print('✗ FAISS not installed')

try:
    import spacy
    nlp = spacy.load('en_core_web_md')
    print('✓ Spacy model loaded')
except:
    print('✗ Spacy model not loaded')

try:
    import fitz
    print('✓ PyMuPDF (PDF support) installed')
except:
    print('⚠ PyMuPDF not installed (PDF support disabled)')

try:
    from docx import Document
    print('✓ python-docx (DOCX support) installed')
except:
    print('⚠ python-docx not installed (DOCX support disabled)')

print('\n✓ Installation verification complete!')
"
```

### 14. Start the Application

```bash
# Start the FastAPI server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 15. Test the Installation

Open your browser and navigate to:

- **API Documentation:** http://localhost:8000/docs
- **System Info:** http://localhost:8000/system/info
- **Health Check:** http://localhost:8000/health

Or use curl:

```bash
# Test system info
curl http://localhost:8000/system/info

# Test health check
curl http://localhost:8000/health
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'X'"

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall requirements
pip install -r requirements.txt
```

### Issue: "Ollama connection error"

**Solution:**
```bash
# Check if Ollama is running
ollama list

# If not running, start it
# Windows/macOS: Start Ollama app
# Linux:
ollama serve &

# Test Ollama
ollama run llama3.2:8b-instruct-q4_K_M "Hello"
```

### Issue: "Spacy model not found"

**Solution:**
```bash
# Download the model again
python -m spacy download en_core_web_md

# If that fails, download directly
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl
```

### Issue: "PDF parsing not working"

**Solution:**
```bash
# Install PyMuPDF
pip install pymupdf

# Or use alternative
pip install pdfplumber
```

### Issue: "DOCX parsing not working"

**Solution:**
```bash
# Install python-docx
pip install python-docx
```

### Issue: "Out of memory when running LLM"

**Solution:**
- Use a smaller model:
  ```bash
  ollama pull gemma2:2b
  ```
- Reduce context window in `services/llm.py`:
  ```python
  num_ctx=2048  # Instead of 4096
  ```

### Issue: "Slow response times"

**Solutions:**
1. Use quantized models (q4_K_M)
2. Reduce chunk size in document parser
3. Reduce k (number of retrieved documents)
4. Enable caching
5. Use GPU if available

---

## Platform-Specific Notes

### Windows

- Use PowerShell or Command Prompt
- Ensure Python is added to PATH
- Some packages may require Visual C++ Build Tools
- Ollama runs as a Windows service

### macOS

- Use Terminal or iTerm2
- May need to install Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```
- Ollama runs as a background service

### Linux

- Ensure python3-dev is installed
- May need to install additional system packages:
  ```bash
  sudo apt install build-essential libssl-dev libffi-dev python3-dev
  ```
- Start Ollama manually or as a service

---

## Next Steps

After successful installation:

1. **Upload Documents:** Use `/store_file` endpoint to upload your documents
2. **Build Index:** Documents are automatically indexed
3. **Query System:** Use `/rectify` or `/analytics` endpoints
4. **Explore API:** Visit http://localhost:8000/docs for interactive API documentation

---

## Getting Help

If you encounter issues not covered here:

1. Check the main README.md
2. Review API documentation at `/docs`
3. Check Ollama documentation: https://ollama.ai/docs
4. Check LangChain documentation: https://python.langchain.com/

---

**Installation complete! 🎉 You're ready to use Sky-Sentinel!**


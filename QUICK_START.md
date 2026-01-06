# Quick Start Guide - Sky-Sentinel

## 🚀 Get Running in 10 Minutes

### Prerequisites
- Python 3.10+
- 8GB RAM
- 10GB free space

### Installation (5 minutes)

```bash
# 1. Clone and setup
git clone <repo-url>
cd Sky-Sentinal
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_md

# 3. Install Ollama
# Download from: https://ollama.ai
# Then pull model:
ollama pull llama3.2:8b-instruct-q4_K_M

# 4. Start server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### First Query (2 minutes)

```bash
# 1. Upload a document
curl -X POST "http://localhost:8000/store_file" \
  -F "file=@your_document.pdf" \
  -F "pb_number=TEST001"

# 2. Query the system
curl -X POST "http://localhost:8000/rectify" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Snag: Your query here",
    "file_name": "default",
    "pb_number": "TEST001"
  }'
```

### Access API Documentation

Open browser: http://localhost:8000/docs

---

## 📋 Key Features

✅ **Multi-Format Support:** PDF, DOCX, TXT, Excel
✅ **Offline Operation:** No internet required
✅ **Citations:** Precise source attribution
✅ **Anti-Hallucination:** Verified, grounded responses
✅ **Incremental Learning:** Add documents without rebuild
✅ **RBAC:** Role-based access control

---

## 🎯 Common Use Cases

### 1. Technical Manual Search
```bash
POST /rectify
{
  "query": "Snag: Hydraulic pressure low",
  "file_name": "default",
  "pb_number": "PROJ001"
}
```

### 2. Analytics
```bash
POST /analytics
{
  "query": "Analyze engine failures",
  "file_name": "default",
  "pb_number": "PROJ001"
}
```

### 3. Verify Query Quality
```bash
POST /verify_query
{
  "query": "Your query here"
}
```

---

## 🔧 Configuration

### Change LLM Model

Edit `services/llm.py`:
```python
model = "llama3.2:8b-instruct-q4_K_M"  # Default
# Or use: qwen2.5:7b-instruct-q4_K_M
```

### Adjust Chunk Size

Edit `services/document_parser.py`:
```python
chunk_size = 1000  # Default
chunk_overlap = 200  # Default
```

---

## 📚 Documentation

- **Full README:** [README.md](README.md)
- **Installation Guide:** [INSTALLATION.md](INSTALLATION.md)
- **Hackathon Guide:** [HACKATHON_GUIDE.md](HACKATHON_GUIDE.md)
- **API Docs:** http://localhost:8000/docs

---

## 🆘 Troubleshooting

**Issue:** Ollama connection error
```bash
# Solution: Start Ollama
ollama serve &
ollama list
```

**Issue:** Module not found
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt
```

**Issue:** Spacy model error
```bash
# Solution: Download model
python -m spacy download en_core_web_md
```

---

## 🎉 You're Ready!

Visit http://localhost:8000/docs to explore all API endpoints.

For detailed documentation, see [README.md](README.md).


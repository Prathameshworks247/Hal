# Admin API Guide - Global Index Management

## 🎯 Overview

The Admin API endpoints allow you to manage the global FAISS index that serves as the default knowledge base for all queries. This enables:

1. **Centralized Knowledge Base**: Build a global index from multiple documents
2. **Incremental Learning**: Add new documents without rebuilding
3. **Batch Processing**: Ingest entire directories at once
4. **Index Management**: View statistics and rebuild when needed

---

## 🔐 Admin Endpoints

### 1. Ingest Single File

**Endpoint:** `POST /admin/ingest/file`

**Description:** Add a single document to the global index

**Parameters:**
- `file` (file): Document to ingest (PDF, DOCX, TXT, XLS, XLSX)
- `incremental` (boolean, optional): Add to existing index (default: true)
- `index_path` (string, optional): Index location (default: "snag_faiss_index")

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/admin/ingest/file" \
  -F "file=@aircraft_manual.pdf" \
  -F "incremental=true" \
  -F "index_path=snag_faiss_index"
```

**Example (Node.js):**
```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const form = new FormData();
form.append('file', fs.createReadStream('aircraft_manual.pdf'));
form.append('incremental', 'true');
form.append('index_path', 'snag_faiss_index');

const response = await axios.post('http://localhost:8000/admin/ingest/file', form, {
  headers: form.getHeaders()
});

console.log(response.data);
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully ingested aircraft_manual.pdf",
  "details": {
    "success": true,
    "file_name": "aircraft_manual.pdf",
    "file_type": ".pdf",
    "num_chunks": 145,
    "index_path": "snag_faiss_index",
    "incremental": true
  }
}
```

---

### 2. Ingest Directory

**Endpoint:** `POST /admin/ingest/directory`

**Description:** Batch ingest all supported files from a directory

**Parameters:**
- `directory_path` (string): Path to directory
- `recursive` (boolean, optional): Include subdirectories (default: true)
- `index_path` (string, optional): Index location (default: "snag_faiss_index")

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/admin/ingest/directory" \
  -F "directory_path=/path/to/documents" \
  -F "recursive=true" \
  -F "index_path=snag_faiss_index"
```

**Example (Node.js):**
```javascript
const axios = require('axios');
const FormData = require('form-data');

const form = new FormData();
form.append('directory_path', '/path/to/documents');
form.append('recursive', 'true');
form.append('index_path', 'snag_faiss_index');

const response = await axios.post('http://localhost:8000/admin/ingest/directory', form, {
  headers: form.getHeaders()
});

console.log(response.data);
```

**Response:**
```json
{
  "success": true,
  "message": "Processed 25 files successfully",
  "details": {
    "success": true,
    "total_files": 27,
    "processed": 25,
    "failed": 2,
    "files": [
      {
        "success": true,
        "file_name": "manual1.pdf",
        "num_chunks": 120
      },
      // ... more files
    ]
  }
}
```

---

### 3. Rebuild Index

**Endpoint:** `POST /admin/index/rebuild`

**Description:** Rebuild entire index from scratch (WARNING: Deletes existing index)

**Parameters:**
- `source_paths` (array of strings): List of file/directory paths
- `index_path` (string, optional): Index location (default: "snag_faiss_index")

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/admin/index/rebuild" \
  -F "source_paths=/path/to/data" \
  -F "source_paths=/path/to/manuals" \
  -F "index_path=snag_faiss_index"
```

**Example (Node.js):**
```javascript
const axios = require('axios');
const FormData = require('form-data');

const form = new FormData();
form.append('source_paths', '/path/to/data');
form.append('source_paths', '/path/to/manuals');
form.append('index_path', 'snag_faiss_index');

const response = await axios.post('http://localhost:8000/admin/index/rebuild', form, {
  headers: form.getHeaders()
});

console.log(response.data);
```

**Response:**
```json
{
  "success": true,
  "message": "Index rebuilt with 1250 documents",
  "details": {
    "success": true,
    "total_documents": 1250,
    "processed_files": 45,
    "failed_files": 2,
    "index_path": "snag_faiss_index"
  }
}
```

---

### 4. Get Index Info

**Endpoint:** `GET /admin/index/info`

**Description:** Get statistics and information about the global index

**Parameters:**
- `index_path` (query param, optional): Index location (default: "snag_faiss_index")

**Example (curl):**
```bash
curl "http://localhost:8000/admin/index/info?index_path=snag_faiss_index"
```

**Example (Node.js):**
```javascript
const axios = require('axios');

const response = await axios.get('http://localhost:8000/admin/index/info', {
  params: {
    index_path: 'snag_faiss_index'
  }
});

console.log(response.data);
```

**Response:**
```json
{
  "success": true,
  "index_info": {
    "exists": true,
    "statistics": {
      "total_documents": 1250,
      "num_sources": 45,
      "created_at": "2026-01-04T10:00:00",
      "last_updated": "2026-01-04T18:30:00",
      "version": 1,
      "num_updates": 12
    },
    "sources": [
      "/path/to/manual1.pdf",
      "/path/to/manual2.pdf",
      // ... more sources
    ],
    "num_sources": 45
  }
}
```

---

## 🚀 Usage Workflow

### Initial Setup (First Time)

```bash
# 1. Rebuild index from your data directory
curl -X POST "http://localhost:8000/admin/index/rebuild" \
  -F "source_paths=data/" \
  -F "source_paths=manuals/"

# 2. Verify index was created
curl "http://localhost:8000/admin/index/info"
```

### Adding New Documents (Incremental)

```bash
# Add single file
curl -X POST "http://localhost:8000/admin/ingest/file" \
  -F "file=@new_manual.pdf" \
  -F "incremental=true"

# Add directory of files
curl -X POST "http://localhost:8000/admin/ingest/directory" \
  -F "directory_path=new_documents/" \
  -F "recursive=true"
```

### Monitoring

```bash
# Check index statistics
curl "http://localhost:8000/admin/index/info"
```

---

## 📊 Node.js Backend Integration

### Complete Example

```javascript
const express = require('express');
const axios = require('axios');
const FormData = require('form-data');
const multer = require('multer');
const fs = require('fs');

const app = express();
const upload = multer({ dest: 'uploads/' });

const PYTHON_API_URL = 'http://localhost:8000';

// Admin: Ingest single file
app.post('/api/admin/ingest/file', upload.single('file'), async (req, res) => {
  try {
    const form = new FormData();
    form.append('file', fs.createReadStream(req.file.path), req.file.originalname);
    form.append('incremental', req.body.incremental || 'true');
    
    const response = await axios.post(`${PYTHON_API_URL}/admin/ingest/file`, form, {
      headers: form.getHeaders()
    });
    
    // Clean up uploaded file
    fs.unlinkSync(req.file.path);
    
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Admin: Ingest directory
app.post('/api/admin/ingest/directory', async (req, res) => {
  try {
    const form = new FormData();
    form.append('directory_path', req.body.directory_path);
    form.append('recursive', req.body.recursive || 'true');
    
    const response = await axios.post(`${PYTHON_API_URL}/admin/ingest/directory`, form, {
      headers: form.getHeaders()
    });
    
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Admin: Get index info
app.get('/api/admin/index/info', async (req, res) => {
  try {
    const response = await axios.get(`${PYTHON_API_URL}/admin/index/info`, {
      params: { index_path: req.query.index_path || 'snag_faiss_index' }
    });
    
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Admin: Rebuild index
app.post('/api/admin/index/rebuild', async (req, res) => {
  try {
    const form = new FormData();
    const paths = Array.isArray(req.body.source_paths) 
      ? req.body.source_paths 
      : [req.body.source_paths];
    
    paths.forEach(path => form.append('source_paths', path));
    
    const response = await axios.post(`${PYTHON_API_URL}/admin/index/rebuild`, form, {
      headers: form.getHeaders()
    });
    
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(3000, () => {
  console.log('Node.js backend running on port 3000');
});
```

---

## 🎯 How Global Index Works

### When User Queries Without Uploading File

```json
{
  "query": "What are aircraft construction materials?",
  "file_name": "default",  // Uses global index
  "pb_number": "PROJ001"
}
```

**Process:**
1. System checks `file_name` = "default"
2. Loads global FAISS index (`snag_faiss_index`)
3. Searches across ALL documents in global index
4. Returns answer with citations from relevant documents

### When User Uploads and Queries Specific File

```json
{
  "query": "What are aircraft construction materials?",
  "file_name": "my_manual.pdf",  // Uses specific file
  "pb_number": "PROJ001"
}
```

**Process:**
1. System checks `file_name` = specific file
2. Loads/creates index for that specific file only
3. Searches only in that document
4. Returns answer with citations from that file

---

## 💡 Best Practices

### 1. Initial Setup
- Use `/admin/index/rebuild` to create initial index from all your documents
- This creates the global knowledge base

### 2. Adding New Documents
- Use `/admin/ingest/file` for single documents (incremental)
- Use `/admin/ingest/directory` for batch additions (incremental)
- No need to rebuild entire index

### 3. Monitoring
- Regularly check `/admin/index/info` for statistics
- Monitor number of documents and sources
- Track last update time

### 4. When to Rebuild
- Only rebuild if index is corrupted
- Or if you want to completely reorganize
- Rebuilding is slow - prefer incremental updates

---

## 🔒 Security Considerations

### For Production:
1. **Add Authentication**: Protect admin endpoints with JWT/API keys
2. **Rate Limiting**: Prevent abuse of ingestion endpoints
3. **File Validation**: Verify file types and sizes
4. **Path Validation**: Sanitize directory paths
5. **RBAC**: Use the built-in RBAC system for admin access

### Example with Authentication:
```javascript
// Middleware to protect admin routes
const adminAuth = (req, res, next) => {
  const token = req.headers['authorization'];
  if (!token || !verifyAdminToken(token)) {
    return res.status(403).json({ error: 'Unauthorized' });
  }
  next();
};

app.post('/api/admin/ingest/file', adminAuth, upload.single('file'), async (req, res) => {
  // ... handler code
});
```

---

## 📈 Performance Tips

1. **Batch Processing**: Use directory ingestion for multiple files
2. **Incremental Updates**: Always use incremental=true unless rebuilding
3. **Monitor Index Size**: Large indexes may need more RAM
4. **Chunk Size**: Adjust in document_parser.py if needed
5. **Background Jobs**: Run large ingestions in background

---

## 🎉 Summary

The admin API provides complete control over your global knowledge base:

✅ **Ingest** any document format (PDF, DOCX, TXT, Excel)
✅ **Incremental learning** - add without rebuilding
✅ **Batch processing** - ingest entire directories
✅ **Statistics** - monitor index health
✅ **Flexible** - users can query global index or specific files

**Result:** A powerful, centralized knowledge base that grows over time!







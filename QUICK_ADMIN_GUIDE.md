# Quick Admin Guide - Global Index Management

## 🎯 What's New

Your system now has **admin endpoints** to manage a global FAISS index that works with **all file formats** (PDF, DOCX, TXT, Excel)!

## 🚀 Quick Start

### 1. Build Initial Global Index

```bash
# From Python directly
python services/ingest.py rebuild data/ manuals/

# Or via API
curl -X POST "http://localhost:8000/admin/index/rebuild" \
  -F "source_paths=data/" \
  -F "source_paths=manuals/"
```

### 2. Add New Documents (Incremental)

```bash
# Single file
curl -X POST "http://localhost:8000/admin/ingest/file" \
  -F "file=@new_manual.pdf" \
  -F "incremental=true"

# Entire directory
curl -X POST "http://localhost:8000/admin/ingest/directory" \
  -F "directory_path=new_documents/" \
  -F "recursive=true"
```

### 3. Check Index Status

```bash
curl "http://localhost:8000/admin/index/info"
```

## 📋 Available Admin Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/admin/ingest/file` | POST | Add single file to index |
| `/admin/ingest/directory` | POST | Add all files from directory |
| `/admin/index/rebuild` | POST | Rebuild entire index |
| `/admin/index/info` | GET | Get index statistics |

## 🔄 How It Works

### Global Index (Default)
When users query with `file_name: "default"`:
```json
{
  "query": "What are aircraft materials?",
  "file_name": "default",  // ← Uses global index
  "pb_number": "PROJ001"
}
```
→ Searches across **ALL documents** in global index

### Specific File
When users upload and query specific file:
```json
{
  "query": "What are aircraft materials?",
  "file_name": "my_manual.pdf",  // ← Uses only this file
  "pb_number": "PROJ001"
}
```
→ Searches only in that specific document

## 💻 Node.js Integration

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

// Add file to global index
async function addToGlobalIndex(filePath) {
  const form = new FormData();
  form.append('file', fs.createReadStream(filePath));
  form.append('incremental', 'true');
  
  const response = await axios.post(
    'http://localhost:8000/admin/ingest/file',
    form,
    { headers: form.getHeaders() }
  );
  
  return response.data;
}

// Get index stats
async function getIndexStats() {
  const response = await axios.get(
    'http://localhost:8000/admin/index/info'
  );
  
  return response.data;
}

// Usage
await addToGlobalIndex('aircraft_manual.pdf');
const stats = await getIndexStats();
console.log(`Total documents: ${stats.index_info.statistics.total_documents}`);
```

## 📊 Supported File Formats

✅ **PDF** - Technical manuals, reports (with page citations)
✅ **DOCX/DOC** - Word documents (with paragraph citations)
✅ **TXT** - Plain text files (with line citations)
✅ **XLSX/XLS** - Excel spreadsheets (with row citations)

## 🎯 Use Cases

### 1. Build Company Knowledge Base
```bash
# Ingest all company manuals
curl -X POST "http://localhost:8000/admin/ingest/directory" \
  -F "directory_path=/company/manuals" \
  -F "recursive=true"
```

### 2. Add New Manual
```bash
# Add single new manual
curl -X POST "http://localhost:8000/admin/ingest/file" \
  -F "file=@new_aircraft_manual.pdf" \
  -F "incremental=true"
```

### 3. Monitor System
```bash
# Check how many documents are indexed
curl "http://localhost:8000/admin/index/info"
```

## ⚡ Key Features

1. **Incremental Learning** - Add documents without rebuilding
2. **Multi-Format** - PDF, DOCX, TXT, Excel all supported
3. **Fast Queries** - Users query global index or specific files
4. **Citations** - Precise page/paragraph/line/row numbers
5. **Scalable** - Handles thousands of documents

## 🔒 Security Note

For production, add authentication to admin endpoints:

```javascript
// Example: Protect admin routes
app.use('/api/admin/*', authenticateAdmin);
```

## 📖 Full Documentation

See `ADMIN_API_GUIDE.md` for complete API documentation and examples.

---

**Ready to use!** Start building your global knowledge base now! 🚀







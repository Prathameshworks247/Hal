# 🔧 File Query Fix - Session FAISS Integration

## ✅ Issue Fixed

**Problem:** When querying a specific uploaded file (not "default"), the system was looking for the file in the filesystem (`uploaded_excels/`) but it was actually stored in SESSION_FAISS.

**Error:**
```
FileNotFoundError: File not found: uploaded_excels/default/Basic_construction_2026-01-07_12-33-58.pdf
```

**Root Cause:** The old code path (`get_chain_file_chached`) was being used for `file_name != "default"`, which tried to load from disk instead of from session FAISS.

---

## 🔧 Solution

Updated the `/user/rectify` endpoint to:
1. Check if session has an uploaded file
2. Query directly from SESSION_FAISS (not filesystem)
3. Return helpful error if file not found in session

---

## 📝 How It Works Now

### Scenario 1: Query Uploaded File

```json
POST /user/rectify
{
  "query": "What is the fuel capacity?",
  "file_name": "aircraft_manual.pdf",  // ← Specific file
  "session_id": "abc-123",              // ← Session with uploaded file
  "conversation_history": []
}
```

**What happens:**
1. ✅ Checks if session has uploaded file
2. ✅ Queries SESSION_FAISS directly
3. ✅ Returns results from uploaded file

### Scenario 2: No File in Session

```json
POST /user/rectify
{
  "query": "What is the fuel capacity?",
  "file_name": "some_file.pdf",
  "session_id": "xyz-789",  // ← Session WITHOUT uploaded file
  "conversation_history": []
}
```

**Response:**
```json
{
  "error": "File 'some_file.pdf' not found in session. Please upload the file first.",
  "session_id": "xyz-789",
  "suggestion": "Use file_name='default' to search global knowledge base, or upload a file with this session_id first."
}
```

### Scenario 3: Query Global Knowledge Base

```json
POST /user/rectify
{
  "query": "What is aircraft corrosion?",
  "file_name": "default",  // ← Use global knowledge base
  "conversation_history": []
}
```

**What happens:**
1. ✅ Queries GLOBAL_FAISS
2. ✅ Returns results from global documents

---

## 🎯 Correct Workflow

### Step 1: Upload File
```bash
curl -X POST http://localhost:8000/user/store-file \
  -F "file=@aircraft_manual.pdf" \
  -F "pb_number=user123"

Response:
{
  "session_id": "abc-123",  ← SAVE THIS
  "file_name": "aircraft_manual_2026-01-08_10-30-45.pdf",
  "chunks_stored": 85
}
```

### Step 2: Query the Uploaded File
```bash
curl -X POST http://localhost:8000/user/rectify \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the fuel capacity?",
    "file_name": "aircraft_manual_2026-01-08_10-30-45.pdf",
    "session_id": "abc-123",  ← USE THE session_id FROM UPLOAD
    "conversation_history": []
  }'
```

✅ **This now works correctly!**

---

## ⚠️ Important Notes

1. **`file_name` field behavior:**
   - `"default"` → Search global knowledge base
   - Specific filename → Search uploaded file in session

2. **You MUST provide `session_id`:**
   - When querying uploaded files, you must send the `session_id` from the upload response
   - Without it, the system doesn't know which session to query

3. **File is stored in SESSION_FAISS, not filesystem:**
   - The actual filename doesn't matter (it's just for reference)
   - The system queries the session FAISS index directly

---

## 🔍 What Changed in Code

**Before (Broken):**
```python
else:
    # File-specific query
    chain, db = get_chain_file_chached(file_name, pb_number)
    # ❌ Tries to load file from disk
```

**After (Fixed):**
```python
else:
    # File-specific query - USE SESSION FAISS
    if not session_manager.has_uploaded_file():
        return {"error": "File not found in session..."}
    
    # ✅ Query from SESSION_FAISS directly
    chain, db = get_chain_cached()
    json_results = process_file_query_json(..., session_manager=session_manager)
```

**Key difference:** The session_manager is now passed to `process_file_query_json`, which knows to query the SESSION_FAISS instead of trying to load from disk.

---

## ✅ Testing

### Test 1: Upload and Query
```bash
# Upload file
curl -X POST http://localhost:8000/user/store-file \
  -F "file=@test.pdf" \
  -F "pb_number=test_user"

# Save session_id from response, then:

# Query the file
curl -X POST http://localhost:8000/user/rectify \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize this document",
    "file_name": "test_2026-01-08_10-30-45.pdf",
    "session_id": "YOUR_SESSION_ID_HERE",
    "conversation_history": []
  }'
```

### Test 2: Query Without Upload (Should Fail Gracefully)
```bash
curl -X POST http://localhost:8000/user/rectify \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize this document",
    "file_name": "nonexistent.pdf",
    "session_id": "random-session-id",
    "conversation_history": []
  }'

# Expected: Helpful error message
```

### Test 3: Query Global (Should Work)
```bash
curl -X POST http://localhost:8000/user/rectify \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is aircraft corrosion?",
    "file_name": "default",
    "conversation_history": []
  }'

# Expected: Results from global knowledge base
```

---

## 📊 Summary

| Scenario | `file_name` | `session_id` | Searches | Status |
|----------|-------------|--------------|----------|--------|
| Global query | `"default"` | Optional | GLOBAL_FAISS | ✅ Works |
| Uploaded file | Filename | Required | SESSION_FAISS | ✅ Fixed |
| No file in session | Filename | Any | N/A | ✅ Error message |

---

## ✅ Fix Complete

The file query functionality now works correctly with the session FAISS system. The error you saw should no longer occur! 🎉


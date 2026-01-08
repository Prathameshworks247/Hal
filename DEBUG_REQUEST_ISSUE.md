# 🔍 Debug: Request Not Processing Issue

## 📊 What I See in the Logs

### Last Request (Line 795-814):
```
📥 Received query request:
   Query: Explain Aircraft diagram
   File: parts-of-an-airplane-9-12 (1)_2026-01-08_18-25-45.pdf
   Session: session_1767877674827
   History: 0 messages

📄 Processing file-specific query for: parts-of-an-airplane-9-12 (1)_2026-01-08_18-25-45.pdf
200 OK
```

## ⚠️ The Problem

**The request IS being processed** (you see 200 OK), but it's likely returning this error:

```json
{
  "error": "File 'parts-of-an-airplane-9-12 (1)_2026-01-08_18-25-45.pdf' not found in session. Please upload the file first.",
  "session_id": "session_1767877674827",
  "suggestion": "Use file_name='default' to search global knowledge base, or upload a file with this session_id first."
}
```

## 🎯 Root Cause

Your Node.js app is sending:
```json
{
  "query": "Explain Aircraft diagram",
  "file_name": "parts-of-an-airplane-9-12 (1)_2026-01-08_18-25-45.pdf",
  "session_id": "session_1767877674827"
}
```

But **this session doesn't have an uploaded file!** 

The session was just created (line 805-806), but no file was uploaded to it.

---

## 🔧 Two Possible Issues

### Issue 1: File Upload Not Working

If the user uploaded a file but it's not in the session, check:

1. **Was the file upload successful?**
   ```bash
   # Check the upload response in your app logs
   # Should see session_id in response
   ```

2. **Are you using the correct session_id?**
   - The session_id from upload response MUST be used in queries
   - Don't generate new session_ids in your app

3. **Check if file was actually embedded:**
   ```bash
   curl http://localhost:8000/admin/sessions
   ```
   Look for your session and check `has_uploaded_file: true`

### Issue 2: Wrong Workflow in Your App

**Current workflow (WRONG):**
```javascript
// User uploads file
const uploadResponse = await uploadFile(file);
// Generates: session_id = "abc-123"

// User queries (WRONG - uses different session)
const queryResponse = await query({
  query: "...",
  file_name: "uploaded_file.pdf",
  session_id: generateNewSessionId()  // ❌ New session!
});
```

**Correct workflow:**
```javascript
// User uploads file
const uploadResponse = await uploadFile(file);
const sessionId = uploadResponse.data.session_id;  // Save this!
const fileName = uploadResponse.data.file_name;    // Save this!

// User queries (CORRECT - uses same session)
const queryResponse = await query({
  query: "...",
  file_name: fileName,        // Use saved filename
  session_id: sessionId       // Use saved session_id
});
```

---

## 🧪 How to Debug

### Step 1: Check Upload Response
Add logging in your Node.js app after file upload:

```javascript
const uploadResponse = await axios.post('/user/store-file', formData);
console.log('Upload response:', uploadResponse.data);
// Should show:
// {
//   session_id: "abc-123",
//   file_name: "...",
//   chunks_stored: 85,
//   storage_location: "session"
// }
```

### Step 2: Check Query Request
Log what you're sending:

```javascript
const payload = {
  query: userInput,
  file_name: savedFileName,
  session_id: savedSessionId
};
console.log('Query payload:', payload);
const response = await axios.post('/user/rectify', payload);
console.log('Query response:', response.data);
```

### Step 3: Check Active Sessions
```bash
curl http://localhost:8000/admin/sessions
```

Look for your session:
```json
{
  "sessions": [
    {
      "session_id": "session_1767877674827",
      "has_uploaded_file": false,  // ← Should be true!
      "uploaded_file_name": null,  // ← Should have filename
      "conversation_turns": 0,
      "total_embeddings": 0        // ← Should be > 0
    }
  ]
}
```

If `has_uploaded_file: false`, the file was never uploaded to this session!

---

## 🎯 Quick Fix Options

### Option 1: Use "default" for Global Search
If you don't need to query a specific uploaded file:

```javascript
const response = await axios.post('/user/rectify', {
  query: userInput,
  file_name: "default",  // ← Search global knowledge base
  conversation_history: []
});
```

### Option 2: Fix Session ID Management
Ensure you're using the SAME session_id from upload:

```javascript
// After upload:
localStorage.setItem('current_session_id', uploadResponse.data.session_id);
localStorage.setItem('current_file_name', uploadResponse.data.file_name);

// When querying:
const sessionId = localStorage.getItem('current_session_id');
const fileName = localStorage.getItem('current_file_name');

const response = await axios.post('/user/rectify', {
  query: userInput,
  file_name: fileName,
  session_id: sessionId,
  conversation_history: []
});
```

---

## 🔍 What Your Response Probably Looks Like

Your app is probably receiving this response:

```json
{
  "error": "File 'parts-of-an-airplane-9-12 (1)_2026-01-08_18-25-45.pdf' not found in session. Please upload the file first.",
  "session_id": "session_1767877674827",
  "suggestion": "Use file_name='default' to search global knowledge base, or upload a file with this session_id first."
}
```

**This is a valid response** (200 OK), but it contains an error field instead of results!

Check your frontend for:
```javascript
if (response.data.error) {
  console.error('Error:', response.data.error);
  console.log('Suggestion:', response.data.suggestion);
}
```

---

## ✅ Action Items

1. **Check if file upload is working**
   - Add console.log after upload
   - Verify session_id and file_name are saved

2. **Verify session_id is being reused**
   - Same session_id from upload should be used in queries
   - Don't generate new session_ids for each query

3. **Check the actual response**
   - Log `response.data` in your app
   - Look for the `error` field

4. **Test with "default"**
   - Try `file_name: "default"` to rule out session issues
   - If this works, it confirms the session management is the problem

---

## 🚀 Quick Test

Run this in your terminal:

```bash
# Test 1: Upload file
curl -X POST http://localhost:8000/user/store-file \
  -F "file=@/path/to/test.pdf" \
  -F "pb_number=test"

# Save the session_id from response, then:

# Test 2: Query with that session_id
curl -X POST http://localhost:8000/user/rectify \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize",
    "file_name": "test_2026-01-08_18-30-00.pdf",
    "session_id": "YOUR_SESSION_ID_HERE",
    "conversation_history": []
  }'
```

If this works, your FastAPI is fine - the issue is in your Node.js session management.

---

## 📝 Summary

**The server IS processing requests** (200 OK status), but it's returning an error response because:

1. The session doesn't have an uploaded file
2. You're querying with a specific filename
3. The system correctly tells you to upload first or use "default"

**Fix:** Ensure your app uses the same session_id from the upload response when querying!


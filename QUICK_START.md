# 🚀 Quick Start Guide - Session FAISS System

## ✅ What Was Fixed

Your FastAPI backend now accepts requests from the Node.js app without requiring `pb_number` in the request body.

---

## 🎯 Changes Summary

### Changed Files:
1. **`models/models.py`** - Made `pb_number` optional
2. **`app.py`** - Added default value handling

### What Node.js Sends:
```json
{
  "query": "Your question here",
  "file_name": "default",
  "session_id": "abc-123",
  "conversation_history": [...]
}
```
✅ No `pb_number` required!

---

## 🧪 Test It Now

### Step 1: Start Server
```bash
cd /Users/prathameshpatil/Sky-Sentinal
source hall/bin/activate
python app.py
```

### Step 2: Run Tests (New Terminal)
```bash
cd /Users/prathameshpatil/Sky-Sentinal
source hall/bin/activate
python test_session_api.py
```

### Step 3: Test with cURL
```bash
# Test 1: New session
curl -X POST http://localhost:8000/user/rectify \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is aircraft corrosion?",
    "file_name": "default",
    "conversation_history": []
  }'

# Save the session_id from response, then:

# Test 2: Follow-up with session
curl -X POST http://localhost:8000/user/rectify \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to prevent it?",
    "file_name": "default",
    "session_id": "YOUR_SESSION_ID_HERE",
    "conversation_history": [
      {"role": "user", "content": "What is aircraft corrosion?"},
      {"role": "assistant", "content": "Corrosion is..."}
    ]
  }'
```

---

## 📊 Expected Behavior

### ✅ What Works Now:

1. **New Query (No session_id)**
   - Backend creates new session automatically
   - Returns session_id in response
   - Retrieves from GLOBAL_FAISS

2. **Follow-up Query (With session_id)**
   - Uses conversation history
   - Retrieves from GLOBAL_FAISS + SESSION_FAISS (conversation memory)
   - Maintains context

3. **File Upload**
   - Creates/reuses session
   - Embeds file in SESSION_FAISS
   - Returns session_id

4. **Query After Upload**
   - Retrieves ONLY from SESSION_FAISS (user's file)
   - Conversation memory included

---

## 🎯 Integration with Node.js

Your Node.js app should:

1. **Send initial query:**
   ```javascript
   const response = await axios.post('/user/rectify', {
     query: userInput,
     file_name: 'default',
     conversation_history: []
   });
   
   // Save session_id
   const sessionId = response.data.session_id;
   ```

2. **Send follow-up queries:**
   ```javascript
   const response = await axios.post('/user/rectify', {
     query: userInput,
     file_name: 'default',
     session_id: sessionId,  // Use saved session_id
     conversation_history: [...history]
   });
   ```

3. **Clean up when done:**
   ```javascript
   await axios.delete(`/user/end-session/${sessionId}`);
   ```

---

## 🔍 Troubleshooting

### Server won't start?
- Check if port 8000 is already in use
- Look for segmentation fault errors (known macOS issue with numpy)

### Validation errors?
- Check that request has `query` and `file_name` (required)
- `session_id` and `conversation_history` are optional

### Sessions not persisting?
- Check `sessions/` directory exists
- Verify session_id is being sent in follow-up requests

---

## 📁 Important Files

| File | Purpose |
|------|---------|
| `models/models.py` | Request/response models |
| `services/session_faiss_manager.py` | Session FAISS logic |
| `services/session_storage.py` | File system operations |
| `test_session_api.py` | Test script |
| `SESSION_FAISS_IMPLEMENTATION.md` | Full documentation |
| `BACKEND_INTEGRATION_FIX.md` | This fix details |

---

## ✅ You're Ready!

Everything is set up. Just:
1. Start the server
2. Run tests to verify
3. Connect your Node.js app

🎉 **The session-based conversational RAG is ready to use!**

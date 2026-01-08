# 🔧 SESSION ID MISMATCH - ROOT CAUSE FOUND

## ✅ Problem Identified

Your Node.js app is sending **different session_ids** for upload vs queries!

---

## 🔍 What I Found

### Upload Session (UUID format):
```
session_id: "48fc7f44-ead5-4ceb-9d04-9eee7d5a0efe"
has_uploaded_file: true
uploaded_file_name: "parts-of-an-airplane-9-12 (1).pdf"
document_chunks: 65
total_embeddings: 66
```
✅ **File is uploaded and embedded here!**

### Query Sessions (MongoDB ObjectId format):
```
session_id: "695fb0c234c2c197e12bb867"
has_uploaded_file: false
uploaded_file_name: null
document_chunks: 0
total_embeddings: 0
```
❌ **Empty session - no file!**

---

## 🎯 The Problem

Your Node.js backend is:

1. **During upload:** Using a UUID as session_id
   - Example: `48fc7f44-ead5-4ceb-9d04-9eee7d5a0efe`
   - FastAPI uploads file to this session ✅

2. **During queries:** Using MongoDB ObjectId as session_id
   - Example: `695fb0c234c2c197e12bb867`
   - FastAPI creates NEW empty session ❌
   - Looks for file in wrong session ❌

---

## 📊 Evidence from Logs

### First request (works):
```
Session: 48fc7f44-ead5-4ceb-9d04-9eee7d5a0efe  ← UUID format
has_uploaded_file: true
```

### Second request onwards (fails):
```
Session: 695fb0c234c2c197e12bb867  ← MongoDB ObjectId!
Created session directory: session-695fb0c234c2c197e12bb867  ← NEW session
has_uploaded_file: false  ← No file!
```

---

## 🔧 The Fix

Your Node.js app needs to **use the SAME session_id format** for both upload and queries.

### Option 1: Use UUID for Everything (Recommended)

```javascript
// In your Node.js backend

// Upload handler
exports.uploadFile = async (req, res) => {
  const formData = new FormData();
  formData.append('file', req.file.buffer, req.file.originalname);
  formData.append('pb_number', req.user.id);
  
  // Let FastAPI generate UUID session_id
  // Don't send session_id in upload request
  
  const response = await axios.post('http://localhost:8000/user/store-file', formData);
  
  const { session_id, file_name } = response.data;
  
  // Save this session_id in MongoDB with the chat
  await Chat.updateOne(
    { _id: req.body.chatId },
    {
      $set: {
        sessionId: session_id,  // ← Save FastAPI's UUID
        fileName: file_name
      }
    }
  );
  
  res.json({ session_id, file_name });
};

// Query handler
exports.sendMessage = async (req, res) => {
  const { chatId, message } = req.body;
  
  // Get the session_id from the chat document
  const chat = await Chat.findById(chatId);
  
  const payload = {
    query: message,
    file_name: chat.fileName || 'default',
    session_id: chat.sessionId,  // ← Use the UUID from FastAPI
    conversation_history: chat.messages
  };
  
  const response = await axios.post('http://localhost:8000/user/rectify', payload);
  res.json(response.data);
};
```

### Option 2: Use MongoDB ObjectId for Everything

```javascript
// Upload handler - send MongoDB ObjectId
exports.uploadFile = async (req, res) => {
  const chatId = req.body.chatId;  // MongoDB ObjectId
  
  const formData = new FormData();
  formData.append('file', req.file.buffer, req.file.originalname);
  formData.append('pb_number', req.user.id);
  formData.append('session_id', chatId.toString());  // ← Send MongoDB ID
  
  const response = await axios.post('http://localhost:8000/user/store-file', formData);
  
  // Save filename in chat
  await Chat.updateOne(
    { _id: chatId },
    { $set: { fileName: response.data.file_name } }
  );
  
  res.json(response.data);
};

// Query handler
exports.sendMessage = async (req, res) => {
  const { chatId, message } = req.body;
  const chat = await Chat.findById(chatId);
  
  const payload = {
    query: message,
    file_name: chat.fileName || 'default',
    session_id: chatId.toString(),  // ← Use same MongoDB ID
    conversation_history: chat.messages
  };
  
  const response = await axios.post('http://localhost:8000/user/rectify', payload);
  res.json(response.data);
};
```

---

## 🎯 Recommended Solution (Option 1)

**Use FastAPI's generated UUID** and store it in MongoDB:

### Your MongoDB Chat Schema:
```javascript
const ChatSchema = new mongoose.Schema({
  userId: String,
  title: String,
  messages: [MessageSchema],
  sessionId: String,  // ← Store FastAPI session_id here
  fileName: String,   // ← Store uploaded filename here
  createdAt: Date,
  updatedAt: Date
});
```

### Upload Flow:
```
1. User uploads file
2. Node.js calls FastAPI /user/store-file (no session_id)
3. FastAPI generates UUID: "48fc7f44-ead5-4ceb-9d04-9eee7d5a0efe"
4. FastAPI embeds file and returns session_id
5. Node.js saves session_id to MongoDB chat document
```

### Query Flow:
```
1. User sends message
2. Node.js gets chat document from MongoDB
3. Node.js reads chat.sessionId (the UUID from FastAPI)
4. Node.js sends query with this session_id
5. FastAPI finds file in correct session
6. Returns results
```

---

## 🧪 Test Your Fix

### Step 1: Upload with Logging
```javascript
console.log('📤 Uploading file...');
const uploadResponse = await axios.post('/user/store-file', formData);
console.log('📥 Upload response:', {
  session_id: uploadResponse.data.session_id,
  file_name: uploadResponse.data.file_name
});

// Save to MongoDB
await Chat.updateOne({ _id: chatId }, {
  $set: {
    sessionId: uploadResponse.data.session_id,
    fileName: uploadResponse.data.file_name
  }
});
console.log('✅ Saved session_id to MongoDB');
```

### Step 2: Query with Logging
```javascript
const chat = await Chat.findById(chatId);
console.log('📤 Sending query with:', {
  session_id: chat.sessionId,
  file_name: chat.fileName
});

const queryResponse = await axios.post('/user/rectify', {
  query: message,
  file_name: chat.fileName,
  session_id: chat.sessionId,  // ← Must match upload!
  conversation_history: chat.messages
});

console.log('📥 Query response status:', queryResponse.data.status);
```

### Expected Output:
```
📤 Uploading file...
📥 Upload response: {
  session_id: '48fc7f44-ead5-4ceb-9d04-9eee7d5a0efe',
  file_name: 'parts-of-an-airplane-9-12_2026-01-08_19-00-00.pdf'
}
✅ Saved session_id to MongoDB

📤 Sending query with: {
  session_id: '48fc7f44-ead5-4ceb-9d04-9eee7d5a0efe',  ← Same UUID!
  file_name: 'parts-of-an-airplane-9-12_2026-01-08_19-00-00.pdf'
}
📥 Query response status: success  ✅
```

---

## 🔍 How to Verify

### Check FastAPI logs:
```bash
tail -f /path/to/fastapi/logs

# Should see:
Using existing session: 48fc7f44-ead5-4ceb-9d04-9eee7d5a0efe  ← Same ID!
Session has uploaded file: true  ✅
```

### Check sessions:
```bash
curl http://localhost:8000/admin/sessions | jq '.sessions[] | select(.session_id == "48fc7f44-ead5-4ceb-9d04-9eee7d5a0efe")'

# Should show:
{
  "session_id": "48fc7f44-ead5-4ceb-9d04-9eee7d5a0efe",
  "has_uploaded_file": true,
  "conversation_turns": 3,  ← Should increase with each query
  "total_embeddings": 69    ← 65 doc + 3 conversations + 1 new conversation
}
```

---

## 📝 Checklist

- [ ] Node.js stores FastAPI's session_id in MongoDB
- [ ] Node.js uses same session_id for queries
- [ ] No new session_ids generated by Node.js
- [ ] Logs show same session_id for upload and queries
- [ ] FastAPI finds file in session (no "file not found" errors)
- [ ] Conversation memory increases with each query

---

## ✅ Summary

**Problem:** Your app uses UUID during upload but MongoDB ObjectId during queries  
**Result:** FastAPI creates new empty sessions for each query  
**Fix:** Use the SAME session_id (FastAPI's UUID) for both upload and queries  
**Action:** Store FastAPI's session_id in your MongoDB chat document  

---

## 🚀 Quick Fix Code

```javascript
// Add this field to your Chat model
sessionId: { type: String, default: null }

// In upload handler
const response = await fastApiUpload(file);
chat.sessionId = response.data.session_id;  // ← Save this!
await chat.save();

// In query handler
const payload = {
  query: message,
  file_name: chat.fileName || 'default',
  session_id: chat.sessionId,  // ← Use the saved one!
  conversation_history: chat.messages
};
```

**This will fix your "no response from 2nd chat onwards" issue!** 🎉


# Intent Classification System

## Overview
The Sky-Sentinel system now automatically classifies user queries into three distinct categories and responds accordingly with specialized formats.

## Intent Categories

### 1. SNAG Intent
**Definition**: Queries describing specific malfunctions, defects, or problems requiring rectification.

**Examples**:
- "Hydraulic pressure low in left landing gear"
- "Engine oil leak detected during pre-flight"
- "APU fails to start"
- "Navigation system showing intermittent errors"

**Response Format**:
1. Problem Analysis
2. Root Cause (if available)
3. Rectification Steps
4. Parts/Tools Required
5. Safety Precautions
6. Source Citations

---

### 2. INSPECTION Intent
**Definition**: Queries asking about inspection procedures, checklists, or preventive maintenance.

**Examples**:
- "How to inspect the main rotor blades?"
- "What are the steps for pre-flight inspection?"
- "Daily inspection checklist for hydraulic system"
- "100-hour inspection procedures for transmission"

**Response Format**:
1. Inspection Overview
2. Inspection Procedures (step-by-step)
3. Tools and Equipment
4. Acceptance Criteria
5. Inspection Frequency
6. Source Citations

---

### 3. CONCEPTUAL Intent
**Definition**: Queries asking for general knowledge, explanations, or theoretical information.

**Examples**:
- "How does the hydraulic system work?"
- "What is the purpose of the pitot tube?"
- "Explain the principle of autorotation"
- "What are the components of the fuel system?"

**Response Format**:
1. Concept Explanation
2. System/Component Description
3. Key Principles
4. Relevant Applications
5. Related Information
6. Source Citations

---

## How It Works

### Automatic Classification
The system automatically analyzes the user's query and determines the intent based on:
- Query structure and keywords
- Context of the question
- Type of information being requested

### Smart Responses
Based on the classified intent, the system:
1. Retrieves relevant information from historical records
2. Structures the response according to the intent type
3. Provides citations and source references
4. Maintains anti-hallucination protocols

---

## Benefits

### For Technicians
- **Faster Troubleshooting**: Snag queries get immediate rectification steps
- **Clear Procedures**: Inspection queries get structured checklists
- **Better Understanding**: Conceptual queries get detailed explanations

### For the System
- **Improved Accuracy**: Intent-specific responses reduce errors
- **Better Context**: Retrieves more relevant historical records
- **Consistent Output**: Standardized format for each intent type

---

## Example Queries & Expected Responses

### Example 1: SNAG Intent
**Query**: "Engine oil pressure low"

**Expected Response**:
```
INTENT: SNAG

Problem Analysis:
According to Record #45, low engine oil pressure typically indicates...

Root Cause:
Based on Record #45 and #67, common causes include...

Rectification Steps:
1. [Step from historical records]
2. [Step from historical records]
...

Parts/Tools Required:
- [Parts mentioned in records]

Safety Precautions:
- [Precautions from records]

Source Citations: Record #45, Record #67
```

### Example 2: INSPECTION Intent
**Query**: "How to inspect landing gear?"

**Expected Response**:
```
INTENT: INSPECTION

Inspection Overview:
According to Page 23, landing gear inspection involves...

Inspection Procedures:
1. [Procedure step from documents]
2. [Procedure step from documents]
...

Tools and Equipment:
- [Tools from documents]

Acceptance Criteria:
- [Criteria from documents]

Source Citations: Page 23, Page 24
```

### Example 3: CONCEPTUAL Intent
**Query**: "How does the hydraulic system work?"

**Expected Response**:
```
INTENT: CONCEPTUAL

Concept Explanation:
According to Page 15, the hydraulic system operates by...

System Description:
The system consists of... [from documents]

Key Principles:
- [Principles from documents]

Relevant Applications:
The hydraulic system is used for... [from documents]

Source Citations: Page 15, Page 16
```

---

## Implementation Details

### Modified Files
- `services/chain_service.py`: Updated `get_chain()` and `get_chain_file()` prompts
- Both functions now include intent classification logic

### Prompt Structure
1. **Step 1**: Intent Classification - Automatically determines query type
2. **Step 2**: Verification Process - Ensures accuracy
3. **Step 3**: Respond According to Intent - Uses appropriate format

### Anti-Hallucination Safeguards
All intent types maintain strict rules:
- Only use information from provided records
- Cite sources for all claims
- State "INSUFFICIENT DATA" when information is missing
- No guessing or fabrication

---

## Usage

### API Endpoints
The intent classification works automatically with existing endpoints:
- `/rectify` - Handles all query types with automatic classification
- `/analytics` - Continues to provide analytical insights

### No Code Changes Required
The system automatically:
1. Receives the query
2. Classifies the intent
3. Retrieves relevant records
4. Formats the response appropriately

---

## Testing

### Test Different Intents
```bash
# Test SNAG intent
curl -X POST "http://localhost:8000/rectify" \
  -H "Content-Type: application/json" \
  -d '{"query": "Engine oil pressure low", "file_name": "default", "pb_number": "TEST"}'

# Test INSPECTION intent
curl -X POST "http://localhost:8000/rectify" \
  -H "Content-Type: application/json" \
  -d '{"query": "How to inspect hydraulic system?", "file_name": "default", "pb_number": "TEST"}'

# Test CONCEPTUAL intent
curl -X POST "http://localhost:8000/rectify" \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain how the fuel pump works", "file_name": "default", "pb_number": "TEST"}'
```

---

## Future Enhancements

Possible improvements:
- Add more intent categories (e.g., DOCUMENTATION, COMPLIANCE)
- Intent confidence scoring
- Mixed intent handling (queries with multiple intents)
- Intent-specific retrieval strategies
- User feedback on intent classification accuracy

---

## Troubleshooting

### Intent Misclassification
If the system misclassifies intent:
- Add more specific keywords to your query
- Rephrase to be more explicit about what you need
- Use indicator words: "inspect", "explain", "fix", "troubleshoot"

### No Response or Empty Response
- Check if historical records contain relevant information
- Verify FAISS index is properly loaded
- Ensure LLM service is running (Ollama with llama3.2)

---

## Notes
- Intent classification is automatic and requires no additional parameters
- All existing API endpoints work seamlessly with the new system
- The system maintains backward compatibility with existing queries
- Citations and traceability are preserved across all intent types


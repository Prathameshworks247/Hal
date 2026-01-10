import logging
from typing import Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from services.llm import get_llm
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pandas as pd
from langchain.schema import Document
from langchain.chains import LLMChain
from services.excel_service import excel_to_documents
from services.document_parser import parse_document
from services.multimodal_embeddings import get_multimodal_embeddings

logger = logging.getLogger(__name__)

def get_chain(department: Optional[str] = None, query: Optional[str] = None):
    try:
        logger.info("Initializing embeddings model...")
        model_path = "./all-MiniLM-L6-v2"
        if not os.path.exists(model_path):
            logger.warning(f"Local model path {model_path} not found.")
            return
        
        embeddings = get_multimodal_embeddings(model_path=model_path, device='cpu')
        logger.info("Multimodal embeddings model loaded successfully.")

        logger.info("Loading FAISS index with department routing...")
        from services.department_routing import route_to_department_faiss
        
        # Route to appropriate department index
        if query is None:
            query = ""
        db, actual_department = route_to_department_faiss(department, query)
        
        if db is None:
            raise FileNotFoundError(f"Failed to load FAISS index for department routing")
        
        logger.info(f"FAISS index loaded successfully. Department: {actual_department}")

        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5}) 
        logger.info("Retriever configured successfully.")

        
        prompt = PromptTemplate.from_template(
        """
You are an expert aircraft technician with extensive experience in aircraft maintenance and troubleshooting.

CRITICAL ANTI-HALLUCINATION RULES:
1. You MUST ONLY use information explicitly present in the provided historical records.
2. You MUST NOT infer, guess, or fabricate any information not directly stated.
3. You MUST reference citations using the provided citation IDs (e.g., "cite_1", "cite_2").
4. If information is missing or unclear, you MUST state "This information is not available in the provided documents" in the content field.
5. You MUST verify each claim against the provided context before including it.

---

CRITICAL OUTPUT REQUIREMENT:
You MUST respond with ONLY valid JSON. No markdown, no explanations, no text outside the JSON structure.
Your entire response must be a valid JSON object matching the exact schema below.

---

CURRENT USER QUERY (ANSWER THIS QUESTION ONLY):
{question}

HISTORICAL RECORDS:
{context}

CRITICAL INSTRUCTION:
- Answer ONLY the CURRENT USER QUERY above. Ignore any previous questions or answers mentioned in the query text.
- Do NOT repeat or summarize information from previous conversation turns.
- If the query asks for a complete list (e.g., "all supervisors", "all names", "all departments"), extract and provide the FULL, COMPLETE list from the HISTORICAL RECORDS above.
- If the query is a follow-up asking for "all" of something, treat it as a NEW request - extract ALL items from the documents, not just what was mentioned before.
- If the query is a COMPARISON request, identify which file version each citation belongs to based on metadata (source_file, file_name, or version information). Clearly indicate in your response which version each piece of information comes from. Compare the content systematically: identify additions, removals, and modifications between versions.
- Base your answer ONLY on the HISTORICAL RECORDS provided above.

---

STEP 1: INTENT CLASSIFICATION
Classify the user's query into one of four categories:
- SNAG: The query describes a specific malfunction, defect, or problem requiring rectification
- INSPECTION: The query asks about inspection procedures, checklists, or preventive maintenance
- CONCEPTUAL: The query asks for general knowledge, explanations, or theoretical information
- COMPARISON: The query asks to compare two versions of uploaded files, identifying differences, additions, removals, or changes between versions

---

STEP 2: CREATE STRUCTURED RESPONSE
Organize your response into sections based on the intent. Each section should have:
- type: One of: "concept_explanation", "system_description", "rectification", "inspection", "problem_analysis", "root_cause", "safety_precautions", "parts_required", "tools_required", "acceptance_criteria", "operational_principles", "related_components", "common_issues", "version_comparison", "added_content", "removed_content", "modified_content", "unchanged_content", "version_details"
- title: A human-readable heading for the section
- content: Clean paragraph text (NO markdown, NO headings, NO bullet points - just plain text)
- citations: Array of citation IDs (e.g., ["cite_1", "cite_2"]) that reference the available citations

---

REQUIRED JSON SCHEMA (respond with ONLY this structure):

{{
  "intent": "SNAG | INSPECTION | CONCEPTUAL | COMPARISON",
  "sections": [
    {{
      "type": "section_type",
      "title": "Human readable heading",
      "content": "Clean paragraph text with no markdown. If information is missing, explicitly state 'This information is not available in the provided documents.'",
      "citations": ["cite_1", "cite_2"]
    }}
  ]
}}

---

SECTION TYPE GUIDELINES:

For SNAG intent, use types like:
- "problem_analysis": Description of the issue
- "root_cause": Most likely cause (if available)
- "rectification": Step-by-step rectification procedures
- "safety_precautions": Safety measures to consider
- "parts_required": Parts that might need replacement

For INSPECTION intent, use types like:
- "inspection": Inspection procedures and scope
- "tools_required": Tools and equipment needed
- "acceptance_criteria": Pass/fail criteria

For CONCEPTUAL intent, use types like:
- "concept_explanation": Explanation/definition
- "system_description": How the system works
- "operational_principles": Key operational principles
- "related_components": Related systems/components
- "common_issues": Typical problems (if applicable)

For COMPARISON intent, use types like:
- "version_details": Information about the file versions being compared (version numbers, dates, filenames)
- "version_comparison": Overall summary of the comparison highlighting key differences
- "added_content": New content, sections, or information added in the newer version
- "removed_content": Content, sections, or information removed from the older version
- "modified_content": Content that was changed, updated, or modified between versions
- "unchanged_content": Content that remained the same across both versions (if relevant)

---

IMPORTANT:
- Do NOT include intent classification text in the content fields
- Do NOT use markdown in content fields
- Do NOT include headings, bullet points, or formatting in content
- Reference citations using citation IDs (e.g., "cite_1", "cite_2") that correspond to the document sources
- Citation IDs should match the format: cite_1, cite_2, cite_3, etc.
- If a section has no information, still include it with content stating "This information is not available in the provided documents"
- Your response must be valid JSON that can be parsed directly
""")
        logger.info("Getting LLM instance...")
        llm = get_llm()
        logger.info("LLM instance obtained successfully.")

        # Create the QA chain with proper input/output keys
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": prompt,
                "verbose": True  # Enable verbose mode for debugging
            },
            return_source_documents=True,  # Return source documents for transparency
            input_key="question",  # Explicitly set input key
            output_key="result"    # Explicitly set output key
        )
        
        logger.info("QA chain created successfully.")
        return qa_chain, db

    except Exception as e:
        logger.error(f"Error in get_chain(): {str(e)}")
        raise


def get_chain_file(file_name, pb_number, use_ocr=False):
    try:
        UPLOAD_DIR = f"uploaded_excels/{pb_number}"
        file_location = os.path.join(UPLOAD_DIR, file_name)
        
        # Use universal document parser for multi-format support
        file_ext = os.path.splitext(file_location)[1].lower()
        if file_ext in ['.xlsx', '.xls']:
            docs = excel_to_documents(file_location)  # Use existing Excel parser for backward compatibility
        else:
            docs = parse_document(file_location, use_ocr=use_ocr)  # Use new multi-format parser with OCR support
        
        print(docs[:20])
        if not docs:
            return {"error": "No relevant historical snag records found."}
        model_path = "./all-MiniLM-L6-v2"
        embeddings = get_multimodal_embeddings(model_path=model_path, device="cpu")

        prompt = PromptTemplate.from_template("""
You are an expert aircraft technician and technical documentation assistant with extensive knowledge of aircraft systems, maintenance, and construction.

CRITICAL ANTI-HALLUCINATION RULES:
1. You MUST ONLY use information explicitly present in the provided documents.
2. You MUST NOT infer, guess, or fabricate any information not directly stated.
3. You MUST reference citations using the provided citation IDs (e.g., "cite_1", "cite_2").
4. If information is missing or unclear, you MUST state "This information is not available in the provided documents" in the content field.
5. You MUST verify each claim against the provided context before including it.

---

CRITICAL OUTPUT REQUIREMENT:
You MUST respond with ONLY valid JSON. No markdown, no explanations, no text outside the JSON structure.
Your entire response must be a valid JSON object matching the exact schema below.

---

CURRENT USER QUESTION (ANSWER THIS QUESTION ONLY):
{question}

RELEVANT DOCUMENT EXCERPTS:
{context}

CRITICAL INSTRUCTION:
- Answer ONLY the CURRENT USER QUESTION above. Ignore any previous questions or answers mentioned in the question text.
- Do NOT repeat or summarize information from previous conversation turns.
- If the question asks for a complete list (e.g., "all supervisors", "all names", "all departments"), extract and provide the FULL, COMPLETE list from the RELEVANT DOCUMENT EXCERPTS above.
- If the question is a follow-up asking for "all" of something, treat it as a NEW request - extract ALL items from the documents, not just what was mentioned before.
- If the question is a COMPARISON request, identify which file version each citation belongs to based on metadata (source_file, file_name, or version information). Clearly indicate in your response which version each piece of information comes from. Compare the content systematically: identify additions, removals, and modifications between versions.
- Base your answer ONLY on the RELEVANT DOCUMENT EXCERPTS provided above.

---

STEP 1: INTENT CLASSIFICATION
Classify the user's query into one of four categories:
- SNAG: The query describes a specific malfunction, defect, or problem requiring rectification
- INSPECTION: The query asks about inspection procedures, checklists, or preventive maintenance
- CONCEPTUAL: The query asks for general knowledge, explanations, or theoretical information
- COMPARISON: The query asks to compare two versions of uploaded files, identifying differences, additions, removals, or changes between versions

---

STEP 2: CREATE STRUCTURED RESPONSE
Organize your response into sections based on the intent. Each section should have:
- type: One of: "concept_explanation", "system_description", "rectification", "inspection", "problem_analysis", "root_cause", "safety_precautions", "parts_required", "tools_required", "acceptance_criteria", "operational_principles", "related_components", "common_issues", "version_comparison", "added_content", "removed_content", "modified_content", "unchanged_content", "version_details"
- title: A human-readable heading for the section
- content: Clean paragraph text (NO markdown, NO headings, NO bullet points - just plain text)
- citations: Array of citation IDs (e.g., ["cite_1", "cite_2"]) that reference the available citations

---

REQUIRED JSON SCHEMA (respond with ONLY this structure):

{{
  "intent": "SNAG | INSPECTION | CONCEPTUAL | COMPARISON",
  "sections": [
    {{
      "type": "section_type",
      "title": "Human readable heading",
      "content": "Clean paragraph text with no markdown. If information is missing, explicitly state 'This information is not available in the provided documents.'",
      "citations": ["cite_1", "cite_2"]
    }}
  ]
}}

---

SECTION TYPE GUIDELINES:

For SNAG intent, use types like:
- "problem_analysis": Description of the issue
- "root_cause": Most likely cause (if available)
- "rectification": Step-by-step rectification procedures
- "safety_precautions": Safety measures to consider
- "parts_required": Parts that might need replacement

For INSPECTION intent, use types like:
- "inspection": Inspection procedures and scope
- "tools_required": Tools and equipment needed
- "acceptance_criteria": Pass/fail criteria

For CONCEPTUAL intent, use types like:
- "concept_explanation": Explanation/definition
- "system_description": How the system works
- "operational_principles": Key operational principles
- "related_components": Related systems/components
- "common_issues": Typical problems (if applicable)

For COMPARISON intent, use types like:
- "version_details": Information about the file versions being compared (version numbers, dates, filenames)
- "version_comparison": Overall summary of the comparison highlighting key differences
- "added_content": New content, sections, or information added in the newer version
- "removed_content": Content, sections, or information removed from the older version
- "modified_content": Content that was changed, updated, or modified between versions
- "unchanged_content": Content that remained the same across both versions (if relevant)

---

IMPORTANT:
- Do NOT include intent classification text in the content fields
- Do NOT use markdown in content fields
- Do NOT include headings, bullet points, or formatting in content
- Reference citations using citation IDs (e.g., "cite_1", "cite_2") that correspond to the document sources
- Citation IDs should match the format: cite_1, cite_2, cite_3, etc.
- If a section has no information, still include it with content stating "This information is not available in the provided documents"
- Your response must be valid JSON that can be parsed directly
""")

        vectorstore = FAISS.from_documents(
            docs,
            embedding=embeddings,
        )

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        logger.info("Getting LLM instance...")
        llm = get_llm()
        logger.info("LLM instance obtained successfully.")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt, "verbose": True},
            return_source_documents=True,
            input_key="question",
            output_key="result"
        )
        return qa_chain,vectorstore
        
    except Exception as e:
        logger.exception("Error during rectification")
        return {"error": str(e)}


def _create_qa_chain_from_db(db):
    """
    Create a QA chain from any FAISS database using the standard prompt.
    Used for creating chains with SESSION_FAISS.
    
    Args:
        db: FAISS vectorstore instance
        
    Returns:
        QA chain instance
    """
    from services.llm import get_llm
    
    prompt = PromptTemplate.from_template("""
You are an expert aircraft technician and technical documentation assistant with extensive knowledge of aircraft systems, maintenance, and construction.

CRITICAL ANTI-HALLUCINATION RULES:
1. You MUST ONLY use information explicitly present in the provided documents.
2. You MUST NOT infer, guess, or fabricate any information not directly stated.
3. You MUST reference citations using the provided citation IDs (e.g., "cite_1", "cite_2").
4. If information is missing or unclear, you MUST state "This information is not available in the provided documents" in the content field.
5. You MUST verify each claim against the provided context before including it.

---

CRITICAL OUTPUT REQUIREMENT:
You MUST respond with ONLY valid JSON. No markdown, no explanations, no text outside the JSON structure.
Your entire response must be a valid JSON object matching the exact schema below.

---

CURRENT USER QUESTION (ANSWER THIS QUESTION ONLY):
{question}

RELEVANT DOCUMENT EXCERPTS:
{context}

CRITICAL INSTRUCTION:
- Answer ONLY the CURRENT USER QUESTION above. Ignore any previous questions or answers mentioned in the question text.
- Do NOT repeat or summarize information from previous conversation turns.
- If the question asks for a complete list (e.g., "all supervisors", "all names", "all departments"), extract and provide the FULL, COMPLETE list from the RELEVANT DOCUMENT EXCERPTS above.
- If the question is a follow-up asking for "all" of something, treat it as a NEW request - extract ALL items from the documents, not just what was mentioned before.
- If the question is a COMPARISON request, identify which file version each citation belongs to based on metadata (source_file, file_name, or version information). Clearly indicate in your response which version each piece of information comes from. Compare the content systematically: identify additions, removals, and modifications between versions.
- Base your answer ONLY on the RELEVANT DOCUMENT EXCERPTS provided above.

---

STEP 1: INTENT CLASSIFICATION
Classify the user's query into one of four categories:
- SNAG: The query describes a specific malfunction, defect, or problem requiring rectification
- INSPECTION: The query asks about inspection procedures, checklists, or preventive maintenance
- CONCEPTUAL: The query asks for general knowledge, explanations, or theoretical information
- COMPARISON: The query asks to compare two versions of uploaded files, identifying differences, additions, removals, or changes between versions

---

STEP 2: CREATE STRUCTURED RESPONSE
Organize your response into sections based on the intent. Each section should have:
- type: One of: "concept_explanation", "system_description", "rectification", "inspection", "problem_analysis", "root_cause", "safety_precautions", "parts_required", "tools_required", "acceptance_criteria", "operational_principles", "related_components", "common_issues", "version_comparison", "added_content", "removed_content", "modified_content", "unchanged_content", "version_details"
- title: A human-readable heading for the section
- content: Clean paragraph text (NO markdown, NO headings, NO bullet points - just plain text)
- citations: Array of citation IDs (e.g., ["cite_1", "cite_2"]) that reference the available citations

---

REQUIRED JSON SCHEMA (respond with ONLY this structure):

{{
  "intent": "SNAG | INSPECTION | CONCEPTUAL | COMPARISON",
  "sections": [
    {{
      "type": "section_type",
      "title": "Human readable heading",
      "content": "Clean paragraph text with no markdown. If information is missing, explicitly state 'This information is not available in the provided documents.'",
      "citations": ["cite_1", "cite_2"]
    }}
  ]
}}

---

SECTION TYPE GUIDELINES:

For SNAG intent, use types like:
- "problem_analysis": Description of the issue
- "root_cause": Most likely cause (if available)
- "rectification": Step-by-step rectification procedures
- "safety_precautions": Safety measures to consider
- "parts_required": Parts that might need replacement

For INSPECTION intent, use types like:
- "inspection": Inspection procedures and scope
- "tools_required": Tools and equipment needed
- "acceptance_criteria": Pass/fail criteria

For CONCEPTUAL intent, use types like:
- "concept_explanation": Explanation/definition
- "system_description": How the system works
- "operational_principles": Key operational principles
- "related_components": Related systems/components
- "common_issues": Typical problems (if applicable)

For COMPARISON intent, use types like:
- "version_details": Information about the file versions being compared (version numbers, dates, filenames)
- "version_comparison": Overall summary of the comparison highlighting key differences
- "added_content": New content, sections, or information added in the newer version
- "removed_content": Content, sections, or information removed from the older version
- "modified_content": Content that was changed, updated, or modified between versions
- "unchanged_content": Content that remained the same across both versions (if relevant)

---

IMPORTANT:
- Do NOT include intent classification text in the content fields
- Do NOT use markdown in content fields
- Do NOT include headings, bullet points, or formatting in content
- Reference citations using citation IDs (e.g., "cite_1", "cite_2") that correspond to the document sources
- Citation IDs should match the format: cite_1, cite_2, cite_3, etc.
- If a section has no information, still include it with content stating "This information is not available in the provided documents"
- Your response must be valid JSON that can be parsed directly
""")
    
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    llm = get_llm()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt, "verbose": True},
        return_source_documents=True,
        input_key="question",
        output_key="result"
    )
    
    return qa_chain
        

def get_analytics_chain():
    try:
        logger.info("Initializing embeddings model...")
        model_path = "./all-MiniLM-L6-v2"
        if not os.path.exists(model_path):
            logger.warning(f"Local model path {model_path} not found.")
            return
        
        embeddings = get_multimodal_embeddings(model_path=model_path, device='cpu')
        logger.info("Multimodal embeddings model loaded successfully.")

        logger.info("Loading FAISS index...")
        # Check if FAISS index exists
        faiss_index_path = "snag_faiss_index"
        if not os.path.exists(faiss_index_path):
            raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")
        
        db = FAISS.load_local(
            faiss_index_path, 
            embeddings=embeddings, 
            allow_dangerous_deserialization=True
        )
        logger.info("FAISS index loaded successfully.")

        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 20}) 
        logger.info("Retriever configured successfully.")

        
        prompt = PromptTemplate.from_template('''
You are an expert aircraft technician and data analyst with deep experience in helicopter maintenance and snag analysis.

CRITICAL ANTI-HALLUCINATION RULES:
1. Extract metrics ONLY from the provided historical records - do not invent numbers.
2. Count actual occurrences in the records - do not estimate or guess.
3. Categories must be derived from actual data patterns in the records.
4. If data is insufficient, use empty objects {{}} rather than making up values.
5. All numbers must be verifiable from the provided context.

---

Given the current snag and historical records, perform an analytical review by:
1. Matching the current snag with similar historical records.
2. Counting actual occurrences and patterns in the records (do not infer beyond what's shown).
3. Creating analytics based ONLY on data present in the historical context.

---
Current Snag:
{question}

Historical Snag Records:
{context}
---

🔧 TASK:
Analyze the current snag using the matched historical records and return only structured analytics in **valid JSON format**.

DATA EXTRACTION INSTRUCTIONS:
- Count actual occurrences in the records - do not estimate.
- Create categories ONLY if you can identify clear patterns in the actual data.
- For metrics (Complexity, Time, Tools, Risk, Frequency), base on patterns in similar records, not guesses.
- If a metric cannot be determined from records, use a conservative default (1-2) or omit it.

VERIFICATION CHECKLIST:
✓ Have I counted actual occurrences from the records?
✓ Are my categories based on actual data patterns?
✓ Can I verify each number from the provided context?
✓ Am I using empty objects {{}} when data is truly insufficient?

IMPORTANT: 
- If the query is not related to the historical snag records, RETURN EXACTLY: {{ "RadarChart": {{}}, "PieChart": {{}}, "BarChart1": {{}}, "BarChart2": {{}} }}
- If data is insufficient for any chart, use empty objects {{}} for that chart.
- Do not include explanations or commentary — just valid JSON.

🎯 OUTPUT FORMAT (use only if you have verifiable data):
```json
{{
  "RadarChart": {{
    "Complexity": 1-5,  // Based on actual complexity indicators in records
    "Time Needed": 1-5,  // Based on time patterns in records
    "Tools Required": 1-5,  // Based on tools mentioned in records
    "Risk Level": 1-5,  // Based on risk indicators in records
    "Frequency": 1-5  // Based on occurrence frequency in records
  }},
  "PieChart": {{
    // Only include categories that appear in the actual records
    "Category-Name": count  // Actual count from records
  }},
  "BarChart1": {{
    // Only include event types that appear in the records
    "Event-Type": count  // Actual count from records
  }},
  "BarChart2": {{
    // Only include aircraft codes that appear in the records
    "Aircraft-Code": count  // Actual count from records
  }}
}}
                                              ''')
        logger.info("Getting LLM instance...")
        llm = get_llm()
        logger.info("LLM instance obtained successfully.")

        # Create the QA chain with proper input/output keys
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": prompt,
                "verbose": True  # Enable verbose mode for debugging
            },
            return_source_documents=True,  # Return source documents for transparency
            input_key="question",  # Explicitly set input key
            output_key="result"    # Explicitly set output key
        )
        
        logger.info("QA chain created successfully.")
        return qa_chain, db

    except Exception as e:
        logger.error(f"Error in get_analytics_chain(): {str(e)}")
        raise
    


def get_analytics_chain_from_xls(file_name, pb_number, use_ocr=False):
    try:
        logger.info("Initializing embeddings model...")
        model_path = "./all-MiniLM-L6-v2"
        if not os.path.exists(model_path):
            logger.warning(f"Local model path {model_path} not found.")
            return

        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cpu'}
        )
        logger.info("Embeddings model loaded successfully.")
        UPLOAD_DIR = f"uploaded_excels/{pb_number}"
        file_location = os.path.join(UPLOAD_DIR, file_name)
        
        # Use universal document parser for multi-format support
        file_ext = os.path.splitext(file_location)[1].lower()
        if file_ext in ['.xlsx', '.xls']:
            docs = excel_to_documents(file_location)  # Use existing Excel parser for backward compatibility
        else:
            docs = parse_document(file_location, use_ocr=use_ocr)  # Use new multi-format parser with OCR support

        logger.info("Building FAISS index in-memory...")
        db = FAISS.from_documents(docs, embedding=embeddings)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 20})
        logger.info("Retriever configured successfully.")

        prompt = PromptTemplate.from_template('''
You are an expert aircraft technician and data analyst with deep experience in helicopter maintenance and snag analysis.

CRITICAL ANTI-HALLUCINATION RULES:
1. Extract metrics ONLY from the provided historical records - do not invent numbers.
2. Count actual occurrences in the records - do not estimate or guess.
3. Categories must be derived from actual data patterns in the records.
4. If data is insufficient, use empty objects {{}} rather than making up values.
5. All numbers must be verifiable from the provided context.

---

Given the current snag and historical records, perform an analytical review by:
1. Matching the current snag with similar historical records.
2. Counting actual occurrences and patterns in the records (do not infer beyond what's shown).
3. Creating analytics based ONLY on data present in the historical context.

---
Current Snag:
{question}

Historical Snag Records:
{context}
---

🔧 TASK:
Analyze the current snag using the matched historical records and return only structured analytics in **valid JSON format**.

DATA EXTRACTION INSTRUCTIONS:
- Count actual occurrences in the records - do not estimate.
- Create categories ONLY if you can identify clear patterns in the actual data.
- For metrics (Complexity, Time, Tools, Risk, Frequency), base on patterns in similar records, not guesses.
- If a metric cannot be determined from records, use a conservative default (1-2) or omit it.

VERIFICATION CHECKLIST:
✓ Have I counted actual occurrences from the records?
✓ Are my categories based on actual data patterns?
✓ Can I verify each number from the provided context?
✓ Am I using empty objects {{}} when data is truly insufficient?

IMPORTANT: 
- If the query is not related to the historical snag records, RETURN EXACTLY: {{ "RadarChart": {{}}, "PieChart": {{}}, "BarChart1": {{}} }}
- If data is insufficient for any chart, use empty objects {{}} for that chart.
- Do not include explanations or commentary — just valid JSON.

🎯 OUTPUT FORMAT (use only if you have verifiable data):
```json
{{
  "RadarChart": {{
    "Complexity": 1-5,  // Based on actual complexity indicators in records
    "Time Needed": 1-5,  // Based on time patterns in records
    "Tools Required": 1-5,  // Based on tools mentioned in records
    "Risk Level": 1-5,  // Based on risk indicators in records
    "Frequency": 1-5  // Based on occurrence frequency in records
  }},
  "PieChart": {{
    // Only include categories that appear in the actual records
    "Category-Name": count  // Actual count from records
  }},
  "BarChart1": {{
    // Only include event types that appear in the records
    "Event-Type": count  // Actual count from records
  }}
}}
        ''')

        logger.info("Getting LLM instance...")
        llm = get_llm()
        logger.info("LLM instance obtained successfully.")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": prompt,
                "verbose": True
            },
            return_source_documents=True,
            input_key="question",
            output_key="result"
        )

        logger.info("QA chain created successfully.")
        return qa_chain, db

    except Exception as e:
        logger.error(f"Error in get_analytics_chain_from_xls(): {str(e)}")
        raise


  
def verify(query):
    try:
        final_query = query 
        logger.info("🔍 Received query for snag verification.")

        prompt = PromptTemplate.from_template("""
You are an expert aircraft technician and data analyst specializing in query validation.

Your task is to determine whether the following input is a valid aircraft snag description or query.

VALIDATION CRITERIA:
A valid aircraft snag/query must:
1. Describe an issue, malfunction, or question related to aircraft components, systems, or operations
2. Contain meaningful technical or operational content
3. Be relevant to aircraft maintenance, troubleshooting, or analysis
4. Not be random text, gibberish, or completely unrelated content

INVALID INPUTS include:
- Random characters or meaningless text
- Completely unrelated topics (e.g., cooking, sports, unrelated technology)
- Inappropriate or offensive content
- Empty or near-empty queries
- Pure numbers or symbols without context

---
Input to validate:
{question}
---

VERIFICATION PROCESS:
1. Does it relate to aircraft, helicopters, or aviation?
2. Does it contain meaningful technical or operational content?
3. Is it a coherent query or description?
4. Is it appropriate and relevant?

Answer with **only** one word: 
- "Yes" if it is a valid aircraft snag description or query
- "No" if it is arbitrary, meaningless, inappropriate, or completely unrelated to aircraft/aviation

Respond with just: Yes or No.
        """)

        llm_chain = LLMChain(
            llm=get_llm(), 
            prompt=prompt,
            verbose=True
        )

        response = llm_chain.invoke({"question": final_query})
        raw_result = response.get("text", "").strip().lower()

        print(raw_result)
        # Optional sanitization
        if "yes" in raw_result:
            return True
        else:
            return False

    except Exception as e:
        logger.error(f"❌ Error in rectification: {e}")
        return {"error": str(e)}

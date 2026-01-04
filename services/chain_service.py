import logging
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from services.llm import get_llm
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pandas as pd
from langchain.schema import Document
import logging
from langchain.chains import LLMChain
from services.excel_service import excel_to_documents
from services.document_parser import parse_document

logger = logging.getLogger(__name__)

def get_chain():
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

        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5}) 
        logger.info("Retriever configured successfully.")

        
        prompt = PromptTemplate.from_template(
        """
You are an expert aircraft technician with extensive experience in aircraft maintenance and troubleshooting.

CRITICAL ANTI-HALLUCINATION RULES:
1. You MUST ONLY use information explicitly present in the provided historical records.
2. You MUST NOT infer, guess, or fabricate any information not directly stated.
3. You MUST cite specific records when making claims (e.g., "Based on Record #X...").
4. If information is missing or unclear, you MUST state "INSUFFICIENT DATA" rather than guessing.
5. You MUST verify each claim against the provided context before including it.

---

You will be provided with:
- A **current aircraft snag report**
- A list of **historical snag records with their corresponding rectifications** (each marked with source information)

Your job is to analyze the current snag **strictly using only the information provided in the historical records**.

---

VERIFICATION PROCESS:
Before responding, verify:
1. Can I find direct evidence for each claim in the provided records?
2. Am I inferring information not explicitly stated? (If yes, remove it)
3. Have I cited the source for each piece of information?

---

CURRENT SNAG:
{question}

HISTORICAL SNAG RECORDS:
{context}

---

RESPONSE FORMAT (follow exactly):

1. **Most Likely Cause of the Issue**
   [State ONLY if directly evident from historical context. Cite source: "Based on Record [X]..."]
   [If not clear, state: "INSUFFICIENT DATA IN HISTORICAL RECORDS TO DETERMINE CAUSE."]

2. **Rectification Suggestions**
   [Mention procedures ONLY if explicitly stated in similar records. Priority based on frequency in records.]
   [Cite: "Suggested based on [X] similar cases in records [list numbers]."]
   [If unclear: "INSUFFICIENT DATA TO PROVIDE SPECIFIC RECTIFICATION SUGGESTIONS."]

3. **Safety Precautions to Consider**
   [Include ONLY if explicitly mentioned in the records. Do not infer safety measures.]
   [If not found: "No specific safety precautions mentioned in historical records."]

4. **Parts That Might Need Replacement**
   [List ONLY parts explicitly mentioned in similar past snags. Include record references.]
   [If none found: "No specific parts identified in historical records."]

---

RECTIFICATION:
[Provide detailed rectification steps STRICTLY based on similar records. For each step, indicate if it's directly from records or inferred (but prefer direct only).]
[If unclear: "INSUFFICIENT DATA IN HISTORICAL RECORDS TO PROVIDE DETAILED RECTIFICATION. Please consult additional technical documentation."]

---

FINAL CHECK: Review your response and remove any information not directly supported by the provided context.
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


def get_chain_file(file_name, pb_number):
    try:
        UPLOAD_DIR = f"uploaded_excels/{pb_number}"
        file_location = os.path.join(UPLOAD_DIR, file_name)
        
        # Use universal document parser for multi-format support
        file_ext = os.path.splitext(file_location)[1].lower()
        if file_ext in ['.xlsx', '.xls']:
            docs = excel_to_documents(file_location)  # Use existing Excel parser for backward compatibility
        else:
            docs = parse_document(file_location)  # Use new multi-format parser
        
        print(docs[:20])
        if not docs:
            return {"error": "No relevant historical snag records found."}
        model_path = "./all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": "cpu"}
        )

        prompt = PromptTemplate.from_template("""
You are an expert aircraft technician with extensive experience in aircraft maintenance and troubleshooting.

CRITICAL ANTI-HALLUCINATION RULES:
1. You MUST ONLY use information explicitly present in the provided historical records.
2. You MUST NOT infer, guess, or fabricate any information not directly stated.
3. You MUST cite specific records when making claims (e.g., "Based on Record #X...").
4. If information is missing or unclear, you MUST state "INSUFFICIENT DATA" rather than guessing.
5. You MUST verify each claim against the provided context before including it.

---

You will be provided with:
- A **current aircraft snag report**
- A list of **historical snag records with their corresponding rectifications** (each marked with source information)

Your job is to analyze the current snag **strictly using only the information provided in the historical records**.

---

VERIFICATION PROCESS:
Before responding, verify:
1. Can I find direct evidence for each claim in the provided records?
2. Am I inferring information not explicitly stated? (If yes, remove it)
3. Have I cited the source for each piece of information?

---

CURRENT SNAG:
{question}

HISTORICAL SNAG RECORDS:
{context}

---

RESPONSE FORMAT (follow exactly):

1. **Most Likely Cause of the Issue**
   [State ONLY if directly evident from historical context. Cite source: "Based on Record [X]..."]
   [If not clear, state: "INSUFFICIENT DATA IN HISTORICAL RECORDS TO DETERMINE CAUSE."]

2. **Rectification Suggestions**
   [Mention procedures ONLY if explicitly stated in similar records. Priority based on frequency in records.]
   [Cite: "Suggested based on [X] similar cases in records [list numbers]."]
   [If unclear: "INSUFFICIENT DATA TO PROVIDE SPECIFIC RECTIFICATION SUGGESTIONS."]

3. **Safety Precautions to Consider**
   [Include ONLY if explicitly mentioned in the records. Do not infer safety measures.]
   [If not found: "No specific safety precautions mentioned in historical records."]

4. **Parts That Might Need Replacement**
   [List ONLY parts explicitly mentioned in similar past snags. Include record references.]
   [If none found: "No specific parts identified in historical records."]

---

RECTIFICATION:
[Provide detailed rectification steps STRICTLY based on similar records. For each step, indicate if it's directly from records or inferred (but prefer direct only).]
[If unclear: "INSUFFICIENT DATA IN HISTORICAL RECORDS TO PROVIDE DETAILED RECTIFICATION. Please consult additional technical documentation."]

---

FINAL CHECK: Review your response and remove any information not directly supported by the provided context.
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
        
    
    

def get_analytics_chain():
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
    


def get_analytics_chain_from_xls(file_name, pb_number):
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
            docs = parse_document(file_location)  # Use new multi-format parser

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

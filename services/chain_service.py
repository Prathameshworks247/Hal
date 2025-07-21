import logger
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

You will be provided with:
- A **current aircraft snag report**
- A list of **historical snag records with their corresponding rectifications**

Your job is to analyze the current snag **strictly using only the information provided in the historical records**.

---

STRICT INSTRUCTIONS:
- DO NOT guess or fabricate information.
- ONLY use data present in the historical context.
- If sufficient information is NOT available in the records, respond with:
  "INSUFFICIENT DATA IN HISTORICAL RECORDS TO ANSWER THIS."

---

CURRENT SNAG:
{question}

HISTORICAL SNAG RECORDS:
{context}

---

Respond strictly in the following format:

1. **Most Likely Cause of the Issue**
   [State only if evident from historical context]

2. **Rectification Suggestions**
   [Mention procedures priority-wise only if it is clearly derivable from context.]

3. **Safety Precautions to Consider**
   [Include only if explicitly present in the records.]

4. **Parts That Might Need Replacement**
   [Only list parts that are mentioned in similar past snags.]
---

RECTIFICATION:
[Provide detailed rectification steps strictly based on similar records. If unclear, state: "INSUFFICIENT DATA IN HISTORICAL RECORDS TO PROVIDE DETAILED RECTIFICATION."]
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
        docs = excel_to_documents(file_location)
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
        
        Based on the following historical snag records and their rectifications, provide a detailed recommendation for fixing the current snag.
        
        Current Snag: {question}
        
        Historical Snag Records:
        {context}
        
        Please provide:
        1. Most likely cause of the issue
        2. Step-by-step rectification procedure
        3. Any safety precautions to consider
        4. Parts that might need replacement
        5. Expected time to complete the fix
        
        Recommended Rectification:
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

Given the current snag and historical records, perform an analytical review by:
1. Matching the current snag with similar historical records.
2. Inferring common patterns and metrics.
3. Creating dynamic analytics including categories and labels from the historical context.

---
Current Snag:
{question}

Historical Snag Records:
{context}
---

🔧 TASK:
Analyze the current snag using the matched historical records and return only structured analytics in **valid JSON format**.

INSTRUCTIONS:
- Create relevant **categories** for the pie chart based on snag types or themes.
- Identify common **event types** (e.g., GR, PLT, Ground Observation) and their no. of occurences in integer for Bar Chart 1.
- Identify common **aircraft types** or codes (e.g., IA, J, ZD, etc.) and their no. of occurences in integer for Bar Chart 2.
- Estimate snag metrics (Complexity, Time, Tools, Risk, Frequency) using a 1-5 scale.
- Do not include explanations or commentary — just valid JSON.

IMPORTANT: If the query is not related to the historical snag records, RETURN EXACTLY this JSON:
{{ "RadarChart": {{}}, "PieChart": {{}}, "BarChart1": {{}}, "BarChart2": {{}} }}

Otherwise, return a valid JSON response in the format below.
🎯 OUTPUT FORMAT:
```json
{{
  "RadarChart": {{
    "Complexity": 1-5,
    "Time Needed": 1-5,
    "Tools Required": 1-5,
    "Risk Level": 1-5,
    "Frequency": 1-5
  }},
  "PieChart": {{
    "Cat-1(insert your category)": 1-100,
    "Cat-2(insert your category)": 1-100,
    "Cat-3(insert your category)": 1-100
  }},
  "BarChart1": {{
    "PLT": int,
    "GR": int,
    "Ground Observation": int
  }},
  "BarChart2": {{
    "IA": int,
    "J": int,
    "ZD": int,
    "IN": int,
    "CG": int
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
        docs = excel_to_documents(file_location)

        logger.info("Building FAISS index in-memory...")
        db = FAISS.from_documents(docs, embedding=embeddings)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 20})
        logger.info("Retriever configured successfully.")

        prompt = PromptTemplate.from_template('''
You are an expert aircraft technician and data analyst with deep experience in helicopter maintenance and snag analysis.

Given the current snag and historical records, perform an analytical review by:
1. Matching the current snag with similar historical records.
2. Inferring common patterns and metrics.
3. Creating dynamic analytics including categories and labels from the historical context.

---
Current Snag:
{question}

Historical Snag Records:
{context}
---

🔧 TASK:
Analyze the current snag using the matched historical records and return only structured analytics in **valid JSON format**.

INSTRUCTIONS:
- Create relevant **categories** for the pie chart based on snag types or themes.
- Identify common **event** types and their no. of occurences in integer for Bar Chart 1.
- Estimate snag metrics (Complexity, Time, Tools, Risk, Frequency) using a 1-5 scale.
- Do not include explanations or commentary — just valid JSON.

IMPORTANT: If the query is not related to the historical snag records, RETURN EXACTLY this JSON:
{{ "RadarChart": {{}}, "PieChart": {{}}, "BarChart1": {{}}}}

Otherwise, return a valid JSON response in the format below.
🎯 OUTPUT FORMAT:
```json
{{
  "RadarChart": {{
    "Complexity": 1-5,
    "Time Needed": 1-5,
    "Tools Required": 1-5,
    "Risk Level": 1-5,
    "Frequency": 1-5
  }},
  "PieChart": {{
    "Cat-1(insert your category)": 1-100,
    "Cat-2(insert your category)": 1-100,
    "Cat-3(insert your category)": 1-100
  }},
  "BarChart1": {{
    "Cat-1(insert some other category)": 1-100,
    "Cat-2(insert some other category)": 1-100,
    "Cat-3(insert some other category)": 1-100
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
        You are an expert aircraft technician and data analyst.

        Your task is to determine whether the following input is a valid aircraft snag description or just random or irrelevant text.

        A valid aircraft snag will typically describe an issue or malfunction in aircraft components or systems, often in clear technical terms.

        ---
        Input:
        {question}
        ---

        Answer with **only** one word: "Yes" if it is a valid snag description, or "No" if it is arbitrary or meaningless or inappropriate or completely unrelated to aircrafts.

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

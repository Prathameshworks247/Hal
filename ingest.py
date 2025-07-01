# ingest_snags_from_excel.py
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os
import logging
def load_snag_excel(file_path):
    df = pd.read_excel(file_path)

    docs = []
    for _, row in df.iterrows():
        event_code = row.get("EVENT", "")
        event_value = (
            "Ground" if event_code == "GR"
            else "Flight" if event_code == "FLT"
            else event_code
        )
        content = f"""
SNAG: {row.get("SNAG_DESCRIPTION", "")}
RECTIFICATION: {row.get("RECTIFICATION", "")}
HELI NO: {row.get("HELI_NO", "")}
HELI HOURS: {row.get("HELI_HOURS", "")}
SNAG DATE: {row.get("SNAG_DATE", "")}
EVENT: {event_value}
RAISED BY: {row.get("RAISED_BY", "")}
STATUS: {row.get("STATUS","")}
"""

        metadata = {
            "helicopter": row.get("HELI_NO", ""),
            "raised_by": row.get("RAISED_BY",""),
            "snag_date": str(row.get("SNAG_DATE", "")),
            "event": event_value,
            "rectified_on": str(row.get("RECTIFICATION_DATE", ""))
        }

        docs.append(Document(page_content=content.strip(), metadata=metadata))

    return docs

def ingest():
    docs = load_snag_excel("All_snags_data.xlsx")
    model_path = "./all-MiniLM-L6-v2"  
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path)
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("snag_faiss_index")

if __name__ == "__main__":
    ingest()

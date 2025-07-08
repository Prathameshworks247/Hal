import pandas as pd
from langchain.docstore.document import Document

def excel_columns(filepath):
    df = pd.read_excel(filepath)
    cols = df.columns
    lis_cols = cols.to_list()
    return lis_cols

def excel_to_documents(file_path: str) -> list[Document]:
    df = pd.read_excel(file_path)
    documents = []

    for idx, row in df.iterrows():
        content_lines = [f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])]
        content = "\n".join(content_lines)

        documents.append(
            Document(
                page_content=content,
                metadata={"row_index": idx, "source": "excel_row"}
            )
        )
    return documents
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
        content_lines = []

        for col in df.columns:
            value = row[col]
            if pd.notna(value):  # Only include non-null fields
                col_clean = str(col).strip().capitalize()
                value_clean = str(value).strip()
                content_lines.append(f"{col_clean}: {value_clean}")

        if content_lines:
            content = "\n".join(content_lines).lower()  # lowercase for embedding consistency

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "row_index": idx,
                        "source": file_path.split("/")[-1],
                        "columns": list(df.columns)
                    }
                )
            )
    return documents

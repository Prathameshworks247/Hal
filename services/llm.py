# llm.py
from langchain_community.llms import Ollama

def get_llm():
    return Ollama(model="gemma2:2b")  # or any other model you prefer
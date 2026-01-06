# llm.py
from langchain_community.llms import Ollama
import os
import logging

logger = logging.getLogger(__name__)

def get_llm(temperature=0.1, model_name=None):
    """
    Get offline LLM instance with optimized settings to reduce hallucination.
    
    Recommended models (in order of preference):
    1. llama3.2:8b-instruct-q4_K_M - Best balance of quality and speed (8B params)
    2. qwen2.5:7b-instruct-q4_K_M - Excellent reasoning, multilingual (7B params)
    3. deepseek-r1:7b-instruct-q4_K_M - Great for technical content (7B params)
    4. mistral:7b-instruct-v0.3-q4_K_M - Good general purpose (7B params)
    5. gemma2:9b-it-q4_K_M - Google's model, good quality (9B params)
    
    Default: llama3.2:8b-instruct-q4_K_M (best for RAG with citations)
    
    Temperature: 0.1 (low) to reduce hallucination, increase determinism
    """
    # Get model from environment or use default
    model = model_name or os.getenv("OLLAMA_MODEL", "gemma2:2b")
    
    logger.info(f"Initializing Ollama LLM with model: {model}, temperature: {temperature}")
    
    return Ollama(
        model=model,
        temperature=temperature,  # Low temperature for factual, deterministic responses
        num_predict=2048,  # Max tokens
        top_p=0.9,  # Nucleus sampling for better quality
        repeat_penalty=1.1,  # Reduce repetition
        num_ctx=4096,  # Context window
    )
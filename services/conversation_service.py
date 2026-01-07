"""
Conversation context service for multi-turn RAG queries.
Maintains conversation history and generates context-aware queries.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation context for multi-turn queries.
    Combines conversation history with current query for better RAG retrieval.
    """
    
    def __init__(self):
        self.max_history_length = 10  # Keep last 10 exchanges
        
    def format_conversation_history(
        self, 
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Format conversation history into a readable string.
        
        Args:
            conversation_history: List of {role, content} dicts
            
        Returns:
            Formatted conversation string
        """
        if not conversation_history:
            return ""
        
        formatted = []
        for msg in conversation_history[-self.max_history_length:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        
        return "\n".join(formatted)
    
    def generate_contextualized_query(
        self,
        current_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a contextualized query that includes conversation history.
        
        Args:
            current_query: The current user question
            conversation_history: Previous conversation messages
            
        Returns:
            Dict with standalone_query, context_summary, and full_context
        """
        if not conversation_history:
            return {
                "standalone_query": current_query,
                "context_summary": "",
                "full_context": "",
                "has_context": False
            }
        
        # Get recent history
        recent_history = conversation_history[-6:]  # Last 3 exchanges
        
        # Format context
        context_formatted = self.format_conversation_history(recent_history)
        
        # Create standalone query that incorporates context
        # This helps the LLM understand what the user is referring to
        standalone_query = self._create_standalone_query(current_query, recent_history)
        
        # Create context summary for logging
        context_summary = self._summarize_context(recent_history)
        
        return {
            "standalone_query": standalone_query,
            "context_summary": context_summary,
            "full_context": context_formatted,
            "has_context": True,
            "history_length": len(recent_history)
        }
    
    def _create_standalone_query(
        self,
        current_query: str,
        history: List[Dict[str, str]]
    ) -> str:
        """
        Create a standalone query that includes context from history.
        This is used for RAG retrieval.
        """
        # Check for pronouns or references that need context
        referential_words = ["it", "this", "that", "these", "those", "they", "them"]
        needs_context = any(word in current_query.lower().split() for word in referential_words)
        
        if not needs_context or not history:
            return current_query
        
        # Get last user question and assistant response
        last_user_msg = None
        last_assistant_msg = None
        
        for msg in reversed(history):
            if msg.get("role") == "user" and not last_user_msg:
                last_user_msg = msg.get("content", "")
            elif msg.get("role") == "assistant" and not last_assistant_msg:
                last_assistant_msg = msg.get("content", "")
            
            if last_user_msg and last_assistant_msg:
                break
        
        # Combine context with current query
        if last_user_msg:
            # Extract the main topic from previous question
            standalone = f"{current_query} (Context: Previous question was about: {last_user_msg[:100]})"
            return standalone
        
        return current_query
    
    def _summarize_context(self, history: List[Dict[str, str]]) -> str:
        """
        Create a brief summary of conversation context.
        """
        if not history:
            return "No previous context"
        
        user_questions = [msg.get("content", "")[:50] for msg in history if msg.get("role") == "user"]
        
        if len(user_questions) == 1:
            return f"Following up on: {user_questions[0]}"
        else:
            return f"Continuing conversation ({len(user_questions)} previous questions)"
    
    def should_use_conversation_context(
        self,
        current_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> bool:
        """
        Determine if conversation context should be used for this query.
        
        Returns:
            True if context should be used
        """
        if not conversation_history or len(conversation_history) == 0:
            return False
        
        # Always use context if available
        return True


# Global conversation manager instance
_conversation_manager = ConversationManager()


def get_conversation_manager() -> ConversationManager:
    """Get the global conversation manager instance."""
    return _conversation_manager


def process_conversational_query(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Process a query with conversation context.
    
    Args:
        query: Current user query
        conversation_history: Previous conversation messages
        
    Returns:
        Dict with contextualized query and metadata
    """
    manager = get_conversation_manager()
    
    result = manager.generate_contextualized_query(query, conversation_history)
    
    logger.info(f"Conversational query processing:")
    logger.info(f"  - Original query: {query}")
    logger.info(f"  - Has context: {result['has_context']}")
    if result['has_context']:
        logger.info(f"  - Context: {result['context_summary']}")
    
    return result


"""
Enhanced prompt verification service with multi-layer security and quality checks.
Prevents malicious inputs and ensures query quality before passing to LLM.
"""
import logging
import re
from typing import Dict, Any, Tuple
import spacy

logger = logging.getLogger(__name__)

# Load spacy model for semantic analysis
try:
    nlp = spacy.load("en_core_web_md")
    SPACY_AVAILABLE = True
except:
    SPACY_AVAILABLE = False
    logger.warning("Spacy model not available. Some verification features will be limited.")


class PromptVerifier:
    """Multi-layer prompt verification system."""
    
    def __init__(self):
        self.min_length = 3
        self.max_length = 5000
        self.min_semantic_threshold = 0.3
        
        # Malicious patterns to detect
        self.malicious_patterns = [
            r"ignore\s+(previous|above|all)\s+instructions",
            r"disregard\s+(previous|above|all)",
            r"forget\s+(everything|all|previous)",
            r"you\s+are\s+now",
            r"new\s+instructions",
            r"system\s+prompt",
            r"<\s*script",  # XSS attempts
            r"javascript:",
            r"eval\s*\(",
            r"exec\s*\(",
            r"__import__",
            r"subprocess",
            r"os\.system",
        ]
        
        # Inappropriate content patterns (expanded)
        self.inappropriate_patterns = [
            # English profanity
            r"\b(fuck|shit|damn|bitch|ass|asshole|bastard|crap|piss|dick|cock|pussy|cunt|whore|slut)\b",
            # Hindi/Urdu profanity
            r"\b(chutiya|madarchod|bhenchod|bhosdike|gandu|randi|saala|kutte|kamina|harami|lodu|laude)\b",
            # Other common inappropriate terms
            r"\b(stupid|idiot|moron|dumb|retard|gay|fag|nigger|kike)\b",
        ]
    
    def verify(self, query: str, context: str = "aircraft") -> Tuple[bool, str, Dict[str, Any]]:
        """
        Comprehensive query verification.
        
        Args:
            query: User query to verify
            context: Expected context (default: "aircraft")
        
        Returns:
            Tuple of (is_valid, error_message, verification_details)
        """
        verification_details = {
            "length_check": False,
            "malicious_check": False,
            "semantic_check": False,
            "relevance_check": False,
            "quality_score": 0.0
        }
        
        # 1. Length check
        if not query or len(query.strip()) < self.min_length:
            return False, "Query too short. Please provide a meaningful query.", verification_details
        
        if len(query) > self.max_length:
            return False, "Query too long. Please limit to 5000 characters.", verification_details
        
        verification_details["length_check"] = True
        
        # 2. Malicious content check
        is_safe, malicious_msg = self._check_malicious_content(query)
        if not is_safe:
            return False, malicious_msg, verification_details
        
        verification_details["malicious_check"] = True
        
        # 3. Inappropriate content check
        is_appropriate, inappropriate_msg = self._check_inappropriate_content(query)
        if not is_appropriate:
            return False, inappropriate_msg, verification_details
        
        # 4. Semantic meaning check
        has_meaning, semantic_msg = self._check_semantic_meaning(query)
        if not has_meaning:
            return False, semantic_msg, verification_details
        
        verification_details["semantic_check"] = True
        
        # 5. Context relevance check (for aircraft/aviation context)
        is_relevant, relevance_score = self._check_relevance(query, context)
        verification_details["relevance_check"] = is_relevant
        verification_details["relevance_score"] = relevance_score
        
        # 6. STRICT: Block if not relevant to aircraft context
        if not is_relevant and context == "aircraft":
            return False, "Query is not related to aircraft/aviation. Please provide an aircraft maintenance or technical query.", verification_details
        
        # 7. Calculate overall quality score
        quality_score = self._calculate_quality_score(query, verification_details)
        verification_details["quality_score"] = quality_score
        
        return True, "Query verified successfully", verification_details
    
    def _check_malicious_content(self, query: str) -> Tuple[bool, str]:
        """Check for malicious patterns like prompt injection."""
        query_lower = query.lower()
        
        for pattern in self.malicious_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                logger.warning(f"Malicious pattern detected: {pattern}")
                return False, "Query contains potentially malicious content and has been blocked."
        
        return True, ""
    
    def _check_inappropriate_content(self, query: str) -> Tuple[bool, str]:
        """Check for inappropriate or offensive content."""
        query_lower = query.lower()
        
        for pattern in self.inappropriate_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                logger.warning(f"Inappropriate content detected")
                return False, "Query contains inappropriate content. Please rephrase professionally."
        
        return True, ""
    
    def _check_semantic_meaning(self, query: str) -> Tuple[bool, str]:
        """Check if query has semantic meaning."""
        # Basic checks first
        words = query.split()
        if len(words) < 2:
            return False, "Query appears to be too simple. Please provide more context."
        
        # Check if mostly alphanumeric
        if len(query) > 0:
            alpha_ratio = sum(c.isalpha() for c in query) / len(query)
            if alpha_ratio < 0.3:
                return False, "Query appears to be random characters. Please enter a meaningful query."
        
        # Check for single word queries (likely not meaningful)
        if len(words) == 1 and len(query.strip()) < 15:
            return False, "Query is too short. Please provide more context about the issue."
        
        if not SPACY_AVAILABLE:
            return True, ""
        
        # Use spacy for semantic analysis
        doc = nlp(query)
        
        # Check vector norm (semantic richness)
        if doc.vector_norm < self.min_semantic_threshold:
            return False, "Query lacks semantic meaning. Please provide a more descriptive query."
        
        # Check if it has at least one noun or verb
        has_content_word = any(token.pos_ in ['NOUN', 'VERB', 'PROPN'] for token in doc)
        if not has_content_word:
            return False, "Query should contain descriptive words about the issue or topic."
        
        return True, ""
    
    def _check_relevance(self, query: str, context: str) -> Tuple[bool, float]:
        """
        Check if query is relevant to the expected context.
        
        Returns:
            Tuple of (is_relevant, relevance_score)
        """
        if context == "aircraft":
            # Aircraft/aviation related keywords
            aircraft_keywords = [
                'aircraft', 'airplane', 'helicopter', 'heli', 'aviation', 'flight',
                'engine', 'rotor', 'blade', 'hydraulic', 'fuel', 'landing', 'takeoff',
                'maintenance', 'snag', 'rectification', 'repair', 'inspection',
                'cockpit', 'avionics', 'navigation', 'altitude', 'pilot', 'crew',
                'wing', 'tail', 'fuselage', 'propeller', 'turbine', 'system',
                'pressure', 'temperature', 'gauge', 'indicator', 'warning', 'light',
                'brake', 'gear', 'flap', 'aileron', 'rudder', 'elevator',
                'radio', 'transponder', 'radar', 'gps', 'instrument', 'panel',
                'electrical', 'mechanical', 'pneumatic', 'control', 'safety',
                'emergency', 'failure', 'malfunction', 'issue', 'problem', 'defect',
                'material', 'construction', 'structure', 'component', 'part', 'assembly',
                'design', 'specification', 'manual', 'procedure', 'technical', 'diagram',
                'schematic', 'blueprint', 'drawing', 'document', 'report', 'data'
            ]
            
            query_lower = query.lower()
            
            # Count keyword matches
            matches = sum(1 for keyword in aircraft_keywords if keyword in query_lower)
            
            # Calculate relevance score
            relevance_score = min(matches / 2.0, 1.0)  # 2+ matches = 100% relevant
            
            # Check for question/action words
            question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'find', 'get', 'show', 'list', 'explain', 'describe', 'analyze', 'search', 'lookup', 'tell']
            has_question_word = any(word in query_lower for word in question_words)
            
            # Check if "aircraft" is explicitly mentioned
            has_aircraft_mention = any(word in query_lower for word in ['aircraft', 'airplane', 'helicopter', 'heli', 'aviation', 'plane'])
            
            # BALANCED: Require aircraft-related keywords with reasonable thresholds
            if has_aircraft_mention:
                # If aircraft is explicitly mentioned, be more lenient
                is_relevant = relevance_score > 0.25
            elif has_question_word and len(query.split()) >= 4:
                # For questions, require at least minimal relevance
                is_relevant = relevance_score > 0.4
            else:
                # For statements/snags, require clear relevance
                is_relevant = relevance_score > 0.5
            
            return is_relevant, relevance_score
        
        return True, 1.0  # Default: accept all if no specific context
    
    def _calculate_quality_score(self, query: str, verification_details: Dict[str, Any]) -> float:
        """Calculate overall query quality score (0-1)."""
        score = 0.0
        
        # Length score (optimal: 20-500 chars)
        length = len(query)
        if 20 <= length <= 500:
            score += 0.3
        elif 10 <= length < 20 or 500 < length <= 1000:
            score += 0.15
        
        # Semantic check passed
        if verification_details.get("semantic_check"):
            score += 0.3
        
        # Relevance score
        score += verification_details.get("relevance_score", 0) * 0.2
        
        # All checks passed
        if all([
            verification_details.get("length_check"),
            verification_details.get("malicious_check"),
            verification_details.get("semantic_check")
        ]):
            score += 0.2
        
        return min(score, 1.0)


# Global verifier instance
_verifier = None

def get_verifier() -> PromptVerifier:
    """Get or create global verifier instance."""
    global _verifier
    if _verifier is None:
        _verifier = PromptVerifier()
    return _verifier


def verify_prompt(query: str, context: str = "aircraft") -> Tuple[bool, str, Dict[str, Any]]:
    """
    Convenience function for prompt verification.
    
    Args:
        query: User query to verify
        context: Expected context
    
    Returns:
        Tuple of (is_valid, error_message, verification_details)
    """
    verifier = get_verifier()
    return verifier.verify(query, context)


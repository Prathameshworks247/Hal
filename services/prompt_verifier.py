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
        
        self.malicious_patterns = [
            r"ignore\s+(previous|above|all)\s+instructions",
            r"disregard\s+(previous|above|all)",
            r"forget\s+(everything|all|previous)",
            r"you\s+are\s+now",
            r"new\s+instructions",
            r"system\s+prompt",
            r"<\s*script",
            r"javascript:",
            r"eval\s*\(",
            r"exec\s*\(",
            r"__import__",
            r"subprocess",
            r"os\.system",
        ]
        
        self.inappropriate_patterns = [
            r"\b(fuck|shit|damn|bitch|ass|asshole|bastard|crap|piss|dick|cock|pussy|cunt|whore|slut)\b",
            r"\b(chutiya|madarchod|bhenchod|bhosdike|gandu|randi|saala|kutte|kamina|harami|lodu|laude)\b",
            r"\b(stupid|idiot|moron|dumb|retard|gay|fag|nigger|kike)\b",
        ]
    
    def verify(self, query: str, context: str = "aircraft") -> Tuple[bool, str, Dict[str, Any]]:
        verification_details = {
            "length_check": False,
            "malicious_check": False,
            "semantic_check": False,
            "relevance_check": False,
            "relevance_warning": False,
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
        
        # 5. Context relevance check (SOFT)
        is_relevant, relevance_score = self._check_relevance(query, context)
        verification_details["relevance_check"] = is_relevant
        verification_details["relevance_score"] = relevance_score
        
        if not is_relevant and context == "aircraft":
            verification_details["relevance_warning"] = True
            verification_details["relevance_note"] = (
                "Low explicit aircraft keyword relevance. "
                "Proceeding assuming document-anchored or contextual query."
            )
            logger.info("⚠️ Soft relevance warning applied (query allowed).")
        
        # 6. Calculate overall quality score
        quality_score = self._calculate_quality_score(query, verification_details)
        
        # Penalize (but do not block) low relevance
        if verification_details["relevance_warning"]:
            quality_score *= 0.7
        
        verification_details["quality_score"] = round(quality_score, 3)
        
        return True, "Query verified successfully", verification_details
    
    def _check_malicious_content(self, query: str) -> Tuple[bool, str]:
        for pattern in self.malicious_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, "Query contains potentially malicious content and has been blocked."
        return True, ""
    
    def _check_inappropriate_content(self, query: str) -> Tuple[bool, str]:
        for pattern in self.inappropriate_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, "Query contains inappropriate content. Please rephrase professionally."
        return True, ""
    
    def _check_semantic_meaning(self, query: str) -> Tuple[bool, str]:
        words = query.split()
        if len(words) < 2:
            return False, "Query appears to be too simple. Please provide more context."
        
        alpha_ratio = sum(c.isalpha() for c in query) / len(query)
        if alpha_ratio < 0.3:
            return False, "Query appears to be random characters."
        
        if not SPACY_AVAILABLE:
            return True, ""
        
        doc = nlp(query)
        if doc.vector_norm < self.min_semantic_threshold:
            return False, "Query lacks semantic meaning."
        
        if not any(tok.pos_ in ["NOUN", "VERB", "PROPN"] for tok in doc):
            return False, "Query lacks meaningful content words."
        
        return True, ""
    
    def _check_relevance(self, query: str, context: str) -> Tuple[bool, float]:
        if context != "aircraft":
            return True, 1.0
        
        aircraft_keywords = [
            'aircraft','airplane','aviation','engine','wing','diagram','system',
            'inspection','maintenance','snag','component','structure','manual'
        ]
        
        matches = sum(1 for kw in aircraft_keywords if kw in query.lower())
        relevance_score = min(matches / 2.0, 1.0)
        
        return relevance_score >= 0.35, relevance_score
    
    def _calculate_quality_score(self, query: str, details: Dict[str, Any]) -> float:
        score = 0.0
        
        length = len(query)
        if 20 <= length <= 500:
            score += 0.3
        elif length >= 10:
            score += 0.15
        
        if details.get("semantic_check"):
            score += 0.3
        
        score += details.get("relevance_score", 0) * 0.2
        
        if all([
            details.get("length_check"),
            details.get("malicious_check"),
            details.get("semantic_check")
        ]):
            score += 0.2
        
        return min(score, 1.0)


_verifier = None

def get_verifier() -> PromptVerifier:
    global _verifier
    if _verifier is None:
        _verifier = PromptVerifier()
    return _verifier


def verify_prompt(query: str, context: str = "aircraft") -> Tuple[bool, str, Dict[str, Any]]:
    verifier = get_verifier()
    return verifier.verify(query, context)
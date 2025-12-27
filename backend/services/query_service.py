"""
Query Enhancement Service

Uses LLM to enhance user queries for better retrieval:
- Expand with related terms
- Detect user intent (explain, compare, simplify)
- Infer difficulty level
- NEW: Clarify messy queries into 3 clear options
"""

import json
from typing import Optional
from pydantic import BaseModel
from openai import OpenAI

from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)


class EnhancedQuery(BaseModel):
    """Enhanced query with metadata."""
    original: str
    enhanced: str
    intent: str  # "explain", "compare", "summarize", "simplify", "deep_dive"
    detected_difficulty: str  # "beginner", "intermediate", "advanced"
    key_concepts: list[str]
    is_comparison: bool = False


# ============ NEW: Clarify Prompt ============

CLARIFY_PROMPT = """You are a prompt clarifier.

The user typed a messy or unclear query about ML/AI. Rewrite it into 3 clear, well-formed questions.

Rules:
- Keep the user's original intent
- Make it grammatically correct and specific
- Each version should have a slightly different angle
- Keep them simple and natural (not keyword lists)

Output JSON only:
{
    "prompts": [
        "Clear version 1",
        "Clear version 2",
        "Clear version 3"
    ]
}

Examples:

User: "GPT, WHAT IS IT"
{
    "prompts": [
        "What is GPT and how does it work?",
        "Explain the GPT architecture and its key components",
        "What is Generative Pre-trained Transformer and why is it important?"
    ]
}

User: "attention??"
{
    "prompts": [
        "What is the attention mechanism in deep learning?",
        "How does self-attention work in transformers?",
        "Why is attention important in neural networks?"
    ]
}

User: "bert vs gpt idk"
{
    "prompts": [
        "What are the key differences between BERT and GPT?",
        "How do BERT and GPT differ in architecture and training?",
        "When should I use BERT vs GPT for NLP tasks?"
    ]
}

User: "make llm go fast"
{
    "prompts": [
        "How can I speed up LLM inference?",
        "What are the best techniques to optimize LLM performance?",
        "How to reduce latency when running large language models?"
    ]
}

User: "transformer explain"
{
    "prompts": [
        "What is a Transformer model and how does it work?",
        "Explain the Transformer architecture step by step",
        "Why are Transformers better than RNNs for NLP?"
    ]
}

User: "lora??"
{
    "prompts": [
        "What is LoRA and how does it work?",
        "How does Low-Rank Adaptation help with fine-tuning LLMs?",
        "What are the benefits of using LoRA over full fine-tuning?"
    ]
}

User: "how does chatgpt work"
{
    "prompts": [
        "How does ChatGPT work under the hood?",
        "What is the architecture behind ChatGPT?",
        "How was ChatGPT trained and what makes it conversational?"
    ]
}
"""


# ============ Existing Enhancement Prompt ============

ENHANCE_PROMPT = """You are a query enhancement system for an academic paper search engine.

Given a user query about machine learning, AI, or computer science, analyze it and output JSON with:

{
    "enhanced": "expanded query with related technical terms for better search",
    "intent": "one of: explain, compare, summarize, simplify, deep_dive",
    "detected_difficulty": "one of: beginner, intermediate, advanced",
    "key_concepts": ["list", "of", "3-5", "key", "concepts"],
    "is_comparison": true/false
}

Intent Detection Rules:
- "ELI5", "simply", "basics", "beginner", "intro" → intent: "simplify", difficulty: "beginner"
- "Compare X vs Y", "difference between" → intent: "compare", is_comparison: true
- "Deep dive", "in-depth", "technical details" → intent: "deep_dive", difficulty: "advanced"
- "How does X work", "What is X" → intent: "explain"
- "Summarize", "overview", "brief" → intent: "summarize"

Query Enhancement Rules:
- Add related technical terms that would appear in academic papers
- Include synonyms and related concepts
- Keep the enhanced query concise (under 15 words)

Examples:
- "ELI5 attention" → enhanced: "attention mechanism transformer neural network basics introduction"
- "BERT vs GPT" → enhanced: "BERT GPT language model comparison pretraining architecture"
- "How do transformers work" → enhanced: "transformer architecture self-attention encoder decoder mechanism"
"""


class QueryService:
    """
    Enhance user queries for better retrieval.
    
    Features:
    - Query expansion with related terms
    - Intent detection (explain vs compare vs simplify)
    - Difficulty inference from phrasing
    - Key concept extraction
    - NEW: Query clarification into 3 options
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
    
    def clarify_query(self, query: str) -> list[str]:
        """
        Generate 3 clearer versions of user's messy query.
        
        Args:
            query: User's raw query
            
        Returns:
            List of 3 clarified prompts
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": CLARIFY_PROMPT},
                    {"role": "user", "content": query}
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=300
            )
            
            result = json.loads(response.choices[0].message.content)
            prompts = result.get("prompts", [])
            
            if len(prompts) >= 3:
                logger.info(f"Clarified: '{query[:30]}...' → 3 options")
                return prompts[:3]
            else:
                logger.warning(f"Got {len(prompts)} prompts, expected 3")
                return [query, query, query]
                
        except Exception as e:
            logger.error(f"Clarification failed: {e}")
            return [query, query, query]
    
    def enhance_query(self, query: str) -> EnhancedQuery:
        """
        Enhance a user query for better retrieval.
        
        Args:
            query: Original user query
            
        Returns:
            EnhancedQuery with expanded terms and metadata
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": ENHANCE_PROMPT},
                    {"role": "user", "content": query}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=300
            )
            
            result = json.loads(response.choices[0].message.content)
            
            enhanced = EnhancedQuery(
                original=query,
                enhanced=result.get("enhanced", query),
                intent=result.get("intent", "explain"),
                detected_difficulty=result.get("detected_difficulty", "beginner"),
                key_concepts=result.get("key_concepts", []),
                is_comparison=result.get("is_comparison", False)
            )
            
            logger.info(
                f"Enhanced query: '{query[:30]}...' → intent={enhanced.intent}, "
                f"difficulty={enhanced.detected_difficulty}"
            )
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")
            # Return basic enhancement on failure
            return EnhancedQuery(
                original=query,
                enhanced=query,
                intent="explain",
                detected_difficulty="beginner",
                key_concepts=query.split()[:5]
            )
    
    def quick_intent_detection(self, query: str) -> tuple[str, str]:
        """
        Fast intent detection without LLM call.
        
        Returns:
            (intent, difficulty)
        """
        query_lower = query.lower()
        
        # Simplify indicators
        if any(word in query_lower for word in ["eli5", "simple", "basics", "beginner", "intro"]):
            return "simplify", "beginner"
        
        # Comparison indicators
        if any(word in query_lower for word in [" vs ", " versus ", "compare", "difference"]):
            return "compare", "intermediate"
        
        # Deep dive indicators
        if any(word in query_lower for word in ["deep dive", "in-depth", "technical", "advanced"]):
            return "deep_dive", "advanced"
        
        # Summary indicators
        if any(word in query_lower for word in ["summarize", "overview", "brief", "tldr"]):
            return "summarize", "beginner"
        
        # Default
        return "explain", "beginner"


# Singleton instance
_query_service: Optional[QueryService] = None


def get_query_service() -> QueryService:
    """Get singleton QueryService instance."""
    global _query_service
    if _query_service is None:
        _query_service = QueryService()
    return _query_service

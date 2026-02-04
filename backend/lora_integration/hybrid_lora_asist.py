"""
Hybrid LLM Service for ASIST
Routes queries between LoRA fine-tuned model (FMS) and Ollama (general)

Author: Tom Lorenc
Version: 1.0
"""

import os
import requests
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from lora_llm import LoRALLM, get_lora_llm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Enum for model types"""
    LORA_FMS = "fms-lora"
    OLLAMA = "ollama"
    AUTO = "auto"


@dataclass
class LLMResponse:
    """Response from LLM query"""
    response: str
    model_used: str
    tokens_generated: Optional[int] = None
    query_type: Optional[str] = None
    confidence: Optional[float] = None


class HybridLLMService:
    """
    Hybrid LLM Service - Intelligent routing between models
    
    Routes queries to the appropriate model:
    - FMS/SAMM queries -> LoRA fine-tuned model
    - General queries -> Ollama
    
    Features:
    - Automatic query classification
    - Lazy model loading
    - Fallback handling
    - Configurable keyword routing
    """
    
    # FMS-related keywords for routing
    DEFAULT_FMS_KEYWORDS = [
        # Core FMS terms
        'fms', 'foreign military sales', 'samm', 'security assistance',
        'dsca', 'defense security cooperation',
        
        # Documents and processes
        'loa', 'letter of offer', 'letter of acceptance',
        'case designator', 'case identifier', 'fms case',
        'p&a', 'price and availability',
        
        # Organizations
        'implementing agency', 'security cooperation office',
        'sco', 'satfa', 'military department',
        
        # Regulations
        'itar', 'ear', 'export control', 'arms export',
        'technology transfer', 'disclosure', 'third party transfer',
        'golden sentry', 'blue lantern',
        
        # Financial
        'fmf', 'foreign military financing', 'imet',
        'pseudo loa', 'blanket order', 'defined order',
        
        # Programs
        'building partner capacity', 'excess defense articles',
        'eda', 'direct commercial sales', 'dcs'
    ]
    
    def __init__(
        self,
        ollama_url: str = None,
        ollama_model: str = None,
        lora_enabled: bool = None,
        fms_keywords: List[str] = None,
        auto_load_lora: bool = False
    ):
        """
        Initialize the Hybrid LLM Service
        
        Args:
            ollama_url: Ollama API URL
            ollama_model: Ollama model name
            lora_enabled: Whether to enable LoRA model
            fms_keywords: Custom FMS keywords for routing
            auto_load_lora: Load LoRA model on init
        """
        # Configuration from environment or parameters
        self.ollama_url = ollama_url or os.getenv(
            "OLLAMA_URL", 
            "http://localhost:11434"
        )
        self.ollama_model = ollama_model or os.getenv(
            "OLLAMA_MODEL", 
            "llama3.1:8b"
        )
        self.lora_enabled = lora_enabled if lora_enabled is not None else \
            os.getenv("ENABLE_LORA", "true").lower() == "true"
        
        # FMS keywords for routing
        self.fms_keywords = fms_keywords or self.DEFAULT_FMS_KEYWORDS
        
        # Model instances (lazy loaded)
        self._lora_llm: Optional[LoRALLM] = None
        self._ollama_available: Optional[bool] = None
        
        # Stats tracking
        self.stats = {
            "lora_queries": 0,
            "ollama_queries": 0,
            "fallback_count": 0,
            "errors": 0
        }
        
        # Auto-load if requested
        if auto_load_lora and self.lora_enabled:
            self._get_lora_model()
    
    def _is_fms_query(self, query: str) -> bool:
        """
        Determine if a query is FMS-related
        
        Args:
            query: User's query text
            
        Returns:
            True if query matches FMS keywords
        """
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.fms_keywords)
    
    def _get_lora_model(self) -> LoRALLM:
        """Lazy load the LoRA model"""
        if self._lora_llm is None:
            logger.info("Loading LoRA model (first use)...")
            self._lora_llm = get_lora_llm()
        return self._lora_llm
    
    def _check_ollama_available(self) -> bool:
        """Check if Ollama is available"""
        if self._ollama_available is not None:
            return self._ollama_available
            
        try:
            response = requests.get(
                f"{self.ollama_url}/api/tags",
                timeout=5
            )
            self._ollama_available = response.status_code == 200
        except Exception:
            self._ollama_available = False
            
        return self._ollama_available
    
    def _query_ollama(
        self, 
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
        Query Ollama for general queries
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        
        return response.json().get('response', '')
    
    def _query_lora(
        self,
        question: str,
        context: str = None,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
        Query the LoRA fine-tuned model
        
        Args:
            question: User question
            context: Optional RAG context
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        lora = self._get_lora_model()
        return lora.query_fms(
            question,
            context=context,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
    
    def query(
        self,
        question: str,
        context: str = None,
        model_type: ModelType = ModelType.AUTO,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> LLMResponse:
        """
        Query the appropriate model based on question type
        
        Args:
            question: User's question
            context: Optional context from RAG
            model_type: Force specific model or auto-route
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            LLMResponse with response and metadata
        """
        # Determine which model to use
        if model_type == ModelType.AUTO:
            use_lora = self.lora_enabled and self._is_fms_query(question)
            query_type = "fms" if use_lora else "general"
        elif model_type == ModelType.LORA_FMS:
            use_lora = True
            query_type = "fms"
        else:
            use_lora = False
            query_type = "general"
        
        try:
            if use_lora:
                # Use LoRA fine-tuned model
                logger.info(f"Routing to LoRA model (FMS query)")
                response = self._query_lora(
                    question,
                    context=context,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                model_used = ModelType.LORA_FMS.value
                self.stats["lora_queries"] += 1
                
            else:
                # Use Ollama
                logger.info(f"Routing to Ollama (general query)")
                
                # Build prompt with context if provided
                if context:
                    prompt = f"""Context information:
{context}

Question: {question}

Please provide a helpful response based on the context above."""
                else:
                    prompt = question
                
                response = self._query_ollama(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                model_used = ModelType.OLLAMA.value
                self.stats["ollama_queries"] += 1
                
        except Exception as e:
            logger.error(f"Primary model failed: {e}")
            self.stats["errors"] += 1
            
            # Try fallback
            try:
                if use_lora and self._check_ollama_available():
                    # Fall back to Ollama
                    logger.info("Falling back to Ollama...")
                    response = self._query_ollama(
                        question,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    model_used = f"{ModelType.OLLAMA.value} (fallback)"
                    self.stats["fallback_count"] += 1
                else:
                    raise e
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                raise RuntimeError(f"All models failed: {e}, {fallback_error}")
        
        return LLMResponse(
            response=response,
            model_used=model_used,
            query_type=query_type
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "lora_enabled": self.lora_enabled,
            "lora_loaded": self._lora_llm is not None and self._lora_llm.is_loaded(),
            "ollama_url": self.ollama_url,
            "ollama_model": self.ollama_model,
            "ollama_available": self._check_ollama_available(),
            "stats": self.stats
        }
    
    def get_stats(self) -> Dict[str, int]:
        """Get query statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset query statistics"""
        self.stats = {
            "lora_queries": 0,
            "ollama_queries": 0,
            "fallback_count": 0,
            "errors": 0
        }


# ============================================================
# Singleton Instance
# ============================================================

_hybrid_service_instance: Optional[HybridLLMService] = None

def get_hybrid_llm_service(**kwargs) -> HybridLLMService:
    """Get or create singleton HybridLLMService"""
    global _hybrid_service_instance
    
    if _hybrid_service_instance is None:
        _hybrid_service_instance = HybridLLMService(**kwargs)
    
    return _hybrid_service_instance


# ============================================================
# CLI Testing
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Hybrid LLM Service")
    parser.add_argument("--question", "-q", type=str, help="Question to ask")
    parser.add_argument("--model", "-m", choices=['auto', 'lora', 'ollama'], default='auto')
    args = parser.parse_args()
    
    service = get_hybrid_llm_service()
    
    model_type = {
        'auto': ModelType.AUTO,
        'lora': ModelType.LORA_FMS,
        'ollama': ModelType.OLLAMA
    }[args.model]
    
    if args.question:
        result = service.query(args.question, model_type=model_type)
        print(f"\nModel: {result.model_used}")
        print(f"Type: {result.query_type}")
        print(f"\nResponse:\n{result.response}")
    else:
        print("\nHybrid LLM Interactive Mode")
        print("Type 'quit' to exit, 'status' for service status\n")
        
        while True:
            question = input("Question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if question.lower() == 'status':
                print(service.get_status())
                continue
            if not question:
                continue
                
            result = service.query(question, model_type=model_type)
            print(f"\n[{result.model_used}] ({result.query_type})")
            print(f"{result.response}\n")

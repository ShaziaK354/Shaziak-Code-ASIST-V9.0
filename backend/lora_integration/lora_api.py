"""
ASIST LoRA API Endpoints
Flask routes for LoRA and Hybrid LLM integration

Author: Tom Lorenc
Version: 1.0

Add these routes to your existing app.py or import as a blueprint
"""

from flask import Blueprint, request, jsonify
from functools import wraps
import logging
import time
from typing import Optional

# Import LLM services
from lora_llm import get_lora_llm, unload_lora_llm, LoRALLM
from hybrid_llm_service import (
    get_hybrid_llm_service, 
    HybridLLMService, 
    ModelType,
    LLMResponse
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Blueprint for LoRA routes
lora_bp = Blueprint('lora', __name__, url_prefix='/api/lora')

# ============================================================
# Global instances (lazy loaded)
# ============================================================
lora_llm: Optional[LoRALLM] = None
hybrid_service: Optional[HybridLLMService] = None


# ============================================================
# Utility Decorators
# ============================================================

def timing_decorator(f):
    """Add timing information to responses"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        elapsed = time.time() - start_time
        
        # Add timing to JSON response if possible
        if isinstance(result, tuple):
            response, status_code = result
        else:
            response = result
            status_code = 200
            
        if hasattr(response, 'json'):
            data = response.get_json()
            if isinstance(data, dict):
                data['elapsed_time'] = round(elapsed, 3)
                return jsonify(data), status_code
                
        return result
    return decorated_function


def handle_errors(f):
    """Standard error handling wrapper"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"API Error in {f.__name__}: {str(e)}", exc_info=True)
            return jsonify({
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }), 500
    return decorated_function


# ============================================================
# LoRA Direct Endpoints
# ============================================================

@lora_bp.route('/query', methods=['POST'])
@handle_errors
@timing_decorator
def lora_query():
    """
    Query the LoRA fine-tuned FMS model directly
    
    Request Body:
        {
            "question": "What is the FMS case designator format?",
            "context": "Optional RAG context...",
            "max_tokens": 512,
            "temperature": 0.7
        }
    
    Response:
        {
            "success": true,
            "response": "The FMS case designator...",
            "model": "fms-llama-v3-lora",
            "elapsed_time": 2.34
        }
    """
    global lora_llm
    
    # Lazy load model
    if lora_llm is None:
        logger.info("Loading LoRA model (first request)...")
        lora_llm = get_lora_llm()
    
    # Parse request
    data = request.get_json() or {}
    question = data.get('question', '')
    context = data.get('context')
    max_tokens = data.get('max_tokens', 512)
    temperature = data.get('temperature', 0.7)
    
    if not question:
        return jsonify({
            'success': False,
            'error': 'Question is required'
        }), 400
    
    # Generate response
    response = lora_llm.query_fms(
        question,
        context=context,
        max_new_tokens=max_tokens,
        temperature=temperature
    )
    
    return jsonify({
        'success': True,
        'response': response,
        'model': 'fms-llama-v3-lora'
    })


@lora_bp.route('/status', methods=['GET'])
@handle_errors
def lora_status():
    """
    Get LoRA model status
    
    Response:
        {
            "loaded": true,
            "model_info": {...}
        }
    """
    global lora_llm
    
    if lora_llm is None:
        return jsonify({
            'loaded': False,
            'model_info': None
        })
    
    return jsonify({
        'loaded': lora_llm.is_loaded(),
        'model_info': lora_llm.get_model_info()
    })


@lora_bp.route('/load', methods=['POST'])
@handle_errors
@timing_decorator
def lora_load():
    """
    Explicitly load the LoRA model
    
    Request Body (optional):
        {
            "base_model": "meta-llama/Llama-3.1-8B-Instruct",
            "lora_path": "./output/fms-llama-v3"
        }
    """
    global lora_llm
    
    data = request.get_json() or {}
    base_model = data.get('base_model')
    lora_path = data.get('lora_path')
    
    lora_llm = get_lora_llm(
        base_model=base_model,
        lora_path=lora_path,
        force_reload=True
    )
    
    return jsonify({
        'success': True,
        'message': 'Model loaded successfully',
        'model_info': lora_llm.get_model_info()
    })


@lora_bp.route('/unload', methods=['POST'])
@handle_errors
def lora_unload():
    """Unload the LoRA model to free memory"""
    global lora_llm
    
    unload_lora_llm()
    lora_llm = None
    
    return jsonify({
        'success': True,
        'message': 'Model unloaded successfully'
    })


# ============================================================
# Hybrid LLM Endpoints
# ============================================================

@lora_bp.route('/hybrid/query', methods=['POST'])
@handle_errors
@timing_decorator
def hybrid_query():
    """
    Query with automatic routing between LoRA and Ollama
    
    Request Body:
        {
            "question": "What is an FMS Letter of Offer?",
            "context": "Optional context...",
            "model": "auto",  // "auto", "lora", or "ollama"
            "max_tokens": 512,
            "temperature": 0.7
        }
    
    Response:
        {
            "success": true,
            "response": "A Letter of Offer and Acceptance...",
            "model_used": "fms-lora",
            "query_type": "fms",
            "elapsed_time": 1.23
        }
    """
    global hybrid_service
    
    # Lazy load service
    if hybrid_service is None:
        hybrid_service = get_hybrid_llm_service()
    
    # Parse request
    data = request.get_json() or {}
    question = data.get('question', '')
    context = data.get('context')
    model = data.get('model', 'auto')
    max_tokens = data.get('max_tokens', 512)
    temperature = data.get('temperature', 0.7)
    
    if not question:
        return jsonify({
            'success': False,
            'error': 'Question is required'
        }), 400
    
    # Map model string to enum
    model_type = {
        'auto': ModelType.AUTO,
        'lora': ModelType.LORA_FMS,
        'ollama': ModelType.OLLAMA
    }.get(model, ModelType.AUTO)
    
    # Query
    result: LLMResponse = hybrid_service.query(
        question,
        context=context,
        model_type=model_type,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    return jsonify({
        'success': True,
        'response': result.response,
        'model_used': result.model_used,
        'query_type': result.query_type
    })


@lora_bp.route('/hybrid/status', methods=['GET'])
@handle_errors
def hybrid_status():
    """
    Get hybrid service status
    
    Response:
        {
            "lora_enabled": true,
            "lora_loaded": true,
            "ollama_available": true,
            "stats": {...}
        }
    """
    global hybrid_service
    
    if hybrid_service is None:
        hybrid_service = get_hybrid_llm_service()
    
    return jsonify(hybrid_service.get_status())


@lora_bp.route('/hybrid/stats', methods=['GET'])
@handle_errors
def hybrid_stats():
    """Get query statistics"""
    global hybrid_service
    
    if hybrid_service is None:
        return jsonify({})
    
    return jsonify(hybrid_service.get_stats())


@lora_bp.route('/hybrid/stats/reset', methods=['POST'])
@handle_errors
def hybrid_stats_reset():
    """Reset query statistics"""
    global hybrid_service
    
    if hybrid_service is not None:
        hybrid_service.reset_stats()
    
    return jsonify({
        'success': True,
        'message': 'Statistics reset'
    })


# ============================================================
# Health Check
# ============================================================

@lora_bp.route('/health', methods=['GET'])
def lora_health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'asist-lora-api'
    })


# ============================================================
# Integration Helper - Add to existing Flask app
# ============================================================

def register_lora_routes(app):
    """
    Register LoRA routes with existing Flask app
    
    Usage in your app.py:
        from lora_api import register_lora_routes
        register_lora_routes(app)
    """
    app.register_blueprint(lora_bp)
    logger.info("LoRA API routes registered at /api/lora/*")


# ============================================================
# Standalone Flask App (for testing)
# ============================================================

def create_app():
    """Create standalone Flask app for testing"""
    from flask import Flask
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app)
    
    app.register_blueprint(lora_bp)
    
    @app.route('/')
    def index():
        return jsonify({
            'service': 'ASIST LoRA API',
            'version': '1.0',
            'endpoints': {
                'POST /api/lora/query': 'Query LoRA model directly',
                'GET /api/lora/status': 'Get LoRA model status',
                'POST /api/lora/load': 'Load LoRA model',
                'POST /api/lora/unload': 'Unload LoRA model',
                'POST /api/lora/hybrid/query': 'Query with auto-routing',
                'GET /api/lora/hybrid/status': 'Get hybrid service status',
                'GET /api/lora/health': 'Health check'
            }
        })
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5001, debug=True)

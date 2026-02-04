"""
ASIST Main Application with LoRA Integration
Example showing how to integrate LoRA endpoints into existing app.py

Author: Tom Lorenc
Version: 8.0 with LoRA
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# Flask App Setup
# ============================================================

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ============================================================
# Import and Register LoRA Routes
# ============================================================

# Option 1: Register as Blueprint (recommended)
from lora_api import register_lora_routes
register_lora_routes(app)

# This adds the following endpoints:
#   POST /api/lora/query         - Query LoRA model directly
#   GET  /api/lora/status        - Get model status
#   POST /api/lora/load          - Load model
#   POST /api/lora/unload        - Unload model
#   POST /api/lora/hybrid/query  - Auto-routing query
#   GET  /api/lora/hybrid/status - Hybrid service status
#   GET  /api/lora/health        - Health check

# ============================================================
# Alternative: Direct Integration in Existing Routes
# ============================================================

# If you want to use LoRA in your existing SAMM query endpoint:

from hybrid_llm_service import get_hybrid_llm_service, ModelType

@app.route('/api/samm/query', methods=['POST'])
def samm_query():
    """
    SAMM Query endpoint with hybrid LoRA/Ollama support
    """
    try:
        data = request.get_json() or {}
        question = data.get('question', '')
        context = data.get('context', '')  # From your RAG retrieval
        
        if not question:
            return jsonify({'error': 'Question required'}), 400
        
        # Get hybrid service (uses LoRA for FMS, Ollama for general)
        llm_service = get_hybrid_llm_service()
        
        # Query with auto-routing
        result = llm_service.query(
            question=question,
            context=context,
            model_type=ModelType.AUTO  # Auto-detects FMS queries
        )
        
        return jsonify({
            'success': True,
            'answer': result.response,
            'model_used': result.model_used,
            'query_type': result.query_type
        })
        
    except Exception as e:
        logger.error(f"SAMM query error: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================
# Integration with Existing Agents
# ============================================================

# Example: Modify your entity_agent to use LoRA

"""
# In entity_agent.py:

from hybrid_llm_service import get_hybrid_llm_service, ModelType

class EntityAgent:
    def __init__(self):
        self.llm_service = get_hybrid_llm_service()
    
    def process_query(self, query: str, retrieved_context: str = None) -> str:
        # Automatically uses LoRA for FMS queries
        result = self.llm_service.query(
            question=query,
            context=retrieved_context,
            model_type=ModelType.AUTO
        )
        return result.response
"""

# ============================================================
# Your Existing Routes (keep as-is)
# ============================================================

@app.route('/')
def index():
    return jsonify({
        'service': 'ASIST V8.0',
        'status': 'running',
        'lora_enabled': os.getenv('ENABLE_LORA', 'true').lower() == 'true'
    })


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})


# Add your existing routes here...
# @app.route('/api/entities', methods=['GET'])
# @app.route('/api/chat', methods=['POST'])
# etc.


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'true').lower() == 'true'
    
    logger.info(f"Starting ASIST V8.0 with LoRA on port {port}")
    logger.info(f"LoRA enabled: {os.getenv('ENABLE_LORA', 'true')}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )

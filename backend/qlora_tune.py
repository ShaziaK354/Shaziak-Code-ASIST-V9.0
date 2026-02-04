"""
RAG Graph QLoRA System with Ollama Integration
===============================================

A comprehensive Graph RAG system with QLoRA fine-tuning capabilities.
Based on the SAMM Agent architecture with enhancements for:
- QLoRA (Quantized Low-Rank Adaptation) fine-tuning
- N-Hop Graph Traversal (1-3+ hops)
- Hybrid Re-ranking with keyword + embedding + boost scoring
- Knowledge Graph integration
- Vector database (ChromaDB) retrieval
- Ollama LLM integration

Features:
- Graph-based retrieval augmented generation
- QLoRA fine-tuning for domain adaptation
- Multi-hop relationship traversal
- Hybrid scoring for improved retrieval
- Gold standard pattern matching
- Caching and performance optimization

Author: Generated from SAMM Agent v5.9.11
Version: 1.0.0
"""

import os
import json
import time
import re
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from collections import deque, defaultdict
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAG_Graph_QLoRA")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RAGConfig:
    """Configuration for RAG Graph QLoRA system."""
    
    # Ollama settings
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_timeout: int = 200
    
    # Vector DB settings
    vector_db_path: str = "./chroma_db"
    embedding_model: str = "all-MiniLM-L6-v2"
    collection_name: str = "rag_documents"
    
    # QLoRA settings
    qlora_enabled: bool = True
    qlora_r: int = 16  # LoRA rank
    qlora_alpha: int = 32  # LoRA alpha
    qlora_dropout: float = 0.05
    qlora_bits: int = 4  # Quantization bits
    qlora_model_path: str = "./qlora_adapter"
    
    # Re-ranking weights
    embedding_weight: float = 0.5
    keyword_weight: float = 0.3
    boost_weight: float = 0.2
    
    # Boost values
    table_boost: float = 0.4
    figure_boost: float = 0.4
    appendix_boost: float = 0.3
    depth_boost_per_level: float = 0.05
    
    # Retrieval settings
    initial_fetch_count: int = 20
    final_return_count: int = 8
    max_hops: int = 3
    
    # Cache settings
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 1000


# =============================================================================
# KNOWLEDGE GRAPH
# =============================================================================

class KnowledgeGraph:
    """
    JSON-based Knowledge Graph for Graph RAG.
    Supports entity storage, relationship management, and n-hop traversal.
    """
    
    def __init__(self, json_path: str = None, json_data: Dict = None):
        self.entities: Dict[str, Dict] = {}
        self.relationships: List[Dict] = []
        self.authority_chains: Dict = {}
        self.question_mappings: Dict = {}
        self.metadata: Dict = {}
        
        # Indices for fast lookup
        self._entity_by_id: Dict[str, Dict] = {}
        self._entity_by_label: Dict[str, Dict] = {}
        self._relationships_by_source: Dict[str, List[Dict]] = {}
        self._relationships_by_target: Dict[str, List[Dict]] = {}
        self._relationship_graph: Dict[str, List[Dict]] = {}
        
        if json_path:
            self._load_from_file(json_path)
        elif json_data:
            self._load_from_dict(json_data)
        
        self._build_indices()
        logger.info(f"KnowledgeGraph loaded: {len(self.entities)} entities, {len(self.relationships)} relationships")
    
    def _load_from_file(self, json_path: str):
        """Load from JSON file."""
        possible_paths = [
            Path(json_path),
            Path.cwd() / json_path,
        ]
        
        for p in possible_paths:
            if p.exists():
                with open(p, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._load_from_dict(data)
                logger.info(f"Loaded KG from: {p}")
                return
        
        logger.warning(f"KG file not found: {json_path}")
    
    def _load_from_dict(self, data: Dict):
        """Load from dictionary."""
        self.metadata = data.get('metadata', {})
        
        # Flatten entities from categories
        entities_data = data.get('entities', {})
        if isinstance(entities_data, dict):
            for category, category_entities in entities_data.items():
                if isinstance(category_entities, dict):
                    for entity_id, entity_data in category_entities.items():
                        self.entities[entity_id] = {
                            'id': entity_id,
                            'type': entity_data.get('type', 'entity'),
                            'properties': {
                                'label': entity_data.get('label', entity_id),
                                'definition': entity_data.get('definition', ''),
                                'section': entity_data.get('section', ''),
                                'category': category
                            }
                        }
                elif isinstance(category_entities, list):
                    for entity_data in category_entities:
                        entity_id = entity_data.get('id', str(len(self.entities)))
                        self.entities[entity_id] = entity_data
        
        # Load relationships
        for rel in data.get('relationships', []):
            self.relationships.append({
                'source': rel.get('source'),
                'target': rel.get('target'),
                'type': rel.get('type'),
                'description': rel.get('description', ''),
                'section': rel.get('section', ''),
                'weight': rel.get('weight', 5)
            })
        
        self.authority_chains = data.get('authority_chains', {})
        self.question_mappings = data.get('question_mappings', {})
    
    def _build_indices(self):
        """Build lookup indices for fast access."""
        for entity_id, entity in self.entities.items():
            self._entity_by_id[entity_id.lower()] = entity
            label = entity.get('properties', {}).get('label', '').lower()
            if label:
                self._entity_by_label[label] = entity
        
        for rel in self.relationships:
            source = (rel['source'] or '').lower()
            target = (rel['target'] or '').lower()
            
            if source:
                if source not in self._relationships_by_source:
                    self._relationships_by_source[source] = []
                self._relationships_by_source[source].append(rel)
                
                if source not in self._relationship_graph:
                    self._relationship_graph[source] = []
                self._relationship_graph[source].append({
                    'target': target,
                    'type': rel.get('type', 'related_to'),
                    'description': rel.get('description', ''),
                    'section': rel.get('section', '')
                })
            
            if target:
                if target not in self._relationships_by_target:
                    self._relationships_by_target[target] = []
                self._relationships_by_target[target].append(rel)
    
    def find_entity(self, query: str) -> Optional[Dict]:
        """Find entity by name or label."""
        query_lower = query.lower().strip()
        
        if query_lower in self._entity_by_id:
            return self._entity_by_id[query_lower]
        if query_lower in self._entity_by_label:
            return self._entity_by_label[query_lower]
        
        # Partial match
        for entity_id, entity in self._entity_by_id.items():
            if query_lower in entity_id:
                return entity
        
        return None
    
    def get_relationships(self, entity_id: str) -> List[Dict]:
        """Get all relationships for entity."""
        entity_lower = entity_id.lower()
        relationships = []
        
        if entity_lower in self._relationships_by_source:
            relationships.extend(self._relationships_by_source[entity_lower])
        if entity_lower in self._relationships_by_target:
            relationships.extend(self._relationships_by_target[entity_lower])
        
        return relationships
    
    def add_entity(self, entity_id: str, entity_data: Dict):
        """Add a new entity to the graph."""
        self.entities[entity_id] = entity_data
        self._entity_by_id[entity_id.lower()] = entity_data
        label = entity_data.get('properties', {}).get('label', '').lower()
        if label:
            self._entity_by_label[label] = entity_data
    
    def add_relationship(self, source: str, target: str, rel_type: str, **kwargs):
        """Add a new relationship to the graph."""
        rel = {
            'source': source,
            'target': target,
            'type': rel_type,
            **kwargs
        }
        self.relationships.append(rel)
        
        # Update indices
        source_lower = source.lower()
        target_lower = target.lower()
        
        if source_lower not in self._relationships_by_source:
            self._relationships_by_source[source_lower] = []
        self._relationships_by_source[source_lower].append(rel)
        
        if source_lower not in self._relationship_graph:
            self._relationship_graph[source_lower] = []
        self._relationship_graph[source_lower].append({
            'target': target_lower,
            'type': rel_type,
            'description': kwargs.get('description', ''),
            'section': kwargs.get('section', '')
        })
    
    def save(self, path: str):
        """Save knowledge graph to JSON file."""
        data = {
            'metadata': self.metadata,
            'entities': self.entities,
            'relationships': self.relationships,
            'authority_chains': self.authority_chains,
            'question_mappings': self.question_mappings
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"KG saved to: {path}")


# =============================================================================
# N-HOP PATH FINDER
# =============================================================================

class NHopPathFinder:
    """
    N-Hop Path RAG Implementation.
    Finds relationship paths up to N hops using BFS.
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.relationship_graph = knowledge_graph._relationship_graph
    
    def find_nhop_paths(self, entity: str, max_hops: int = 3, max_paths: int = 15) -> List[Dict]:
        """
        Find all n-hop paths from entity using BFS.
        
        Args:
            entity: Starting entity
            max_hops: Maximum number of hops (1, 2, 3, or more)
            max_paths: Maximum paths to return
            
        Returns:
            List of path dictionaries
        """
        entity_lower = entity.lower()
        paths = []
        
        if entity_lower not in self.relationship_graph:
            return paths
        
        # BFS: (current_node, path_nodes, path_relationships, path_sections, depth)
        queue = deque([(entity_lower, [entity_lower], [], [], 0)])
        visited_paths = set()
        
        while queue and len(paths) < max_paths * 2:
            current, path_nodes, path_rels, path_sections, depth = queue.popleft()
            
            if depth >= max_hops:
                continue
            
            if current in self.relationship_graph:
                for edge in self.relationship_graph[current]:
                    target = edge['target']
                    rel_type = edge['type']
                    section = edge.get('section', '')
                    
                    # Avoid cycles
                    if target in path_nodes:
                        continue
                    
                    new_path_nodes = path_nodes + [target]
                    new_path_rels = path_rels + [rel_type]
                    new_path_sections = path_sections + [section]
                    new_depth = depth + 1
                    
                    # Build path text
                    path_text = f"{entity.upper()}"
                    for i, rel in enumerate(new_path_rels):
                        path_text += f" --[{rel}]--> {new_path_nodes[i+1].upper()}"
                    
                    path_key = " -> ".join(new_path_nodes)
                    
                    if path_key not in visited_paths:
                        visited_paths.add(path_key)
                        paths.append({
                            'hops': new_depth,
                            'path': new_path_nodes,
                            'relationships': new_path_rels,
                            'path_text': path_text,
                            'sections': new_path_sections
                        })
                    
                    if new_depth < max_hops:
                        queue.append((target, new_path_nodes, new_path_rels, new_path_sections, new_depth))
        
        paths.sort(key=lambda x: x['hops'])
        return paths[:max_paths]
    
    def find_supervision_chain(self, entity: str) -> List[Dict]:
        """Find supervision/authority chain for entity."""
        entity_lower = entity.lower()
        chain = []
        current = entity_lower
        visited = {current}
        
        supervision_types = ['supervised_by', 'reports_to', 'managed_by', 'directed_by', 'part_of']
        
        for _ in range(5):  # Max 5 levels
            found = False
            if current in self.relationship_graph:
                for edge in self.relationship_graph[current]:
                    if edge['type'] in supervision_types:
                        target = edge['target']
                        if target not in visited:
                            chain.append({
                                'from': current,
                                'to': target,
                                'type': edge['type'],
                                'section': edge.get('section', '')
                            })
                            visited.add(target)
                            current = target
                            found = True
                            break
            
            if not found:
                break
        
        return chain
    
    def get_context_for_query(self, entities: List[str], query: str, intent: str = None) -> Dict:
        """Get n-hop context for a query."""
        result = {
            'paths': [],
            'authority_chains': {},
            'context_text': '',
            'is_authority_question': False,
            'relationship_count': 0
        }
        
        # Check if authority/supervision question
        authority_keywords = ['supervise', 'supervision', 'report', 'oversee', 'manage', 
                            'direct', 'authority', 'responsible', 'who']
        query_lower = query.lower()
        result['is_authority_question'] = any(kw in query_lower for kw in authority_keywords)
        
        all_paths = []
        
        for entity in entities:
            # Get n-hop paths
            paths = self.find_nhop_paths(entity, max_hops=3)
            all_paths.extend(paths)
            
            # Get supervision chain for authority questions
            if result['is_authority_question']:
                chain = self.find_supervision_chain(entity)
                if chain:
                    result['authority_chains'][entity] = chain
        
        # Deduplicate paths
        seen_paths = set()
        unique_paths = []
        for path in all_paths:
            path_key = path['path_text']
            if path_key not in seen_paths:
                seen_paths.add(path_key)
                unique_paths.append(path)
        
        # Sort by relevance
        if result['is_authority_question']:
            supervision_types = ['supervised_by', 'reports_to', 'supervises', 'manages', 'directs']
            unique_paths.sort(key=lambda p: -sum(1 for r in p['relationships'] if r in supervision_types))
        
        result['paths'] = unique_paths[:10]
        result['relationship_count'] = len(unique_paths)
        
        # Build context text
        context_parts = []
        
        if result['is_authority_question'] and result['authority_chains']:
            context_parts.append("=== AUTHORITY/SUPERVISION CHAINS ===")
            for entity, chain in result['authority_chains'].items():
                if chain:
                    chain_text = f"{entity.upper()}"
                    for edge in chain:
                        chain_text += f" --[{edge['type']}]--> {edge['to'].upper()}"
                    context_parts.append(chain_text)
        
        if result['paths']:
            context_parts.append("\n=== RELATIONSHIP PATHS ===")
            for path in result['paths'][:5]:
                context_parts.append(f"• {path['path_text']}")
                if path.get('sections'):
                    sections = [s for s in path['sections'] if s]
                    if sections:
                        context_parts.append(f"  Reference: {', '.join(sections)}")
        
        result['context_text'] = '\n'.join(context_parts)
        
        return result


# =============================================================================
# VECTOR DATABASE
# =============================================================================

class VectorDatabase:
    """
    ChromaDB-based vector database for RAG.
    Handles document embedding and retrieval.
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = None
        self.collection = None
        self.embedding_model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB and embedding model."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.client = chromadb.PersistentClient(
                path=self.config.vector_db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"ChromaDB initialized: {self.config.vector_db_path}")
            
        except ImportError:
            logger.warning("ChromaDB not installed. Run: pip install chromadb")
            self.client = None
        
        try:
            from sentence_transformers import SentenceTransformer
            
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            logger.info(f"Embedding model loaded: {self.config.embedding_model}")
            
        except ImportError:
            logger.warning("SentenceTransformers not installed. Run: pip install sentence-transformers")
            self.embedding_model = None
    
    def add_documents(self, documents: List[Dict], batch_size: int = 100):
        """
        Add documents to the vector database.
        
        Args:
            documents: List of dicts with 'id', 'content', 'metadata'
            batch_size: Number of documents per batch
        """
        if not self.collection or not self.embedding_model:
            logger.error("Vector database not initialized")
            return
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            ids = [doc['id'] for doc in batch]
            contents = [doc['content'] for doc in batch]
            metadatas = [doc.get('metadata', {}) for doc in batch]
            
            embeddings = self.embedding_model.encode(contents).tolist()
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(batch)} documents (batch {i // batch_size + 1})")
    
    def query(self, query: str, n_results: int = None) -> List[Dict]:
        """
        Query the vector database.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of result dictionaries
        """
        if not self.collection or not self.embedding_model:
            logger.error("Vector database not initialized")
            return []
        
        n_results = n_results or self.config.initial_fetch_count
        
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                formatted_results.append({
                    'id': doc_id,
                    'content': results['documents'][0][i] if results['documents'] else '',
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0
                })
        
        return formatted_results
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        if self.collection:
            return self.collection.count()
        return 0


# =============================================================================
# HYBRID RE-RANKER
# =============================================================================

class HybridReranker:
    """
    Hybrid re-ranking system combining:
    - Embedding similarity
    - Keyword matching
    - Table/Figure/Appendix boost
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'to', 'of',
            'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'and', 'but',
            'if', 'or', 'what', 'how', 'when', 'where', 'why', 'who', 'which',
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'am', 'this', 'that'
        }
    
    def calculate_keyword_score(self, query: str, chunk_content: str) -> float:
        """Calculate keyword match score between query and chunk."""
        query_lower = query.lower()
        chunk_lower = chunk_content.lower()
        
        # Extract meaningful words from query
        query_words = set()
        for word in re.findall(r'\b[a-z0-9]+\b', query_lower):
            if word not in self.stopwords and len(word) > 2:
                query_words.add(word)
        
        if not query_words:
            return 0.0
        
        # Count matches
        matches = sum(1 for word in query_words if word in chunk_lower)
        score = matches / len(query_words)
        
        # Bonus for phrase matches
        query_phrases = re.findall(r'\b\w+\s+\w+\b', query_lower)
        for phrase in query_phrases[:3]:
            words = phrase.split()
            if len(words) == 2 and words[0] not in self.stopwords and phrase in chunk_lower:
                score = min(1.0, score + 0.1)
        
        return min(1.0, score)
    
    def calculate_boost_score(self, chunk_content: str, chunk_metadata: Dict) -> float:
        """Calculate boost score for Tables, Figures, Appendix."""
        boost = 0.0
        content_lower = chunk_content.lower()
        
        # Table boost
        if re.search(r'table\s*[a-z]?\d+', content_lower):
            boost += self.config.table_boost
        
        # Figure boost
        if re.search(r'figure\s*[a-z]?\d+', content_lower):
            boost += self.config.figure_boost
        
        # Appendix boost
        if re.search(r'appendix\s+\d+', content_lower):
            boost += self.config.appendix_boost
        
        # Section depth boost
        section_match = re.search(r'[A-Z]?\d+(\.\d+)+', chunk_content)
        if section_match:
            section = section_match.group(0)
            depth = len(section.split('.'))
            depth_boost = min(0.3, (depth - 2) * self.config.depth_boost_per_level)
            boost += max(0, depth_boost)
        
        return boost
    
    def rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Re-rank search results using hybrid scoring.
        
        Combined Score = embedding_weight * embedding_score
                       + keyword_weight * keyword_score
                       + boost_weight * boost_score
        """
        if not results:
            return results
        
        logger.info(f"Re-ranking {len(results)} results...")
        
        scored_results = []
        
        for r in results:
            content = r.get('content', '')
            metadata = r.get('metadata', {})
            
            # 1. Embedding score (convert distance to similarity)
            distance = r.get('distance', 0.5)
            embedding_score = max(0, 1 - distance) if distance <= 1 else 0.5
            
            # 2. Keyword score
            keyword_score = self.calculate_keyword_score(query, content)
            
            # 3. Boost score
            boost_score = self.calculate_boost_score(content, metadata)
            
            # Combined score
            final_score = (
                self.config.embedding_weight * embedding_score +
                self.config.keyword_weight * keyword_score +
                self.config.boost_weight * boost_score
            )
            
            r['_rerank_scores'] = {
                'embedding': round(embedding_score, 3),
                'keyword': round(keyword_score, 3),
                'boost': round(boost_score, 3),
                'final': round(final_score, 3)
            }
            r['_final_score'] = final_score
            scored_results.append(r)
        
        # Sort by final score descending
        scored_results.sort(key=lambda x: x['_final_score'], reverse=True)
        
        return scored_results[:self.config.final_return_count]


# =============================================================================
# QLORA FINE-TUNER
# =============================================================================

class QLoRAFineTuner:
    """
    QLoRA (Quantized Low-Rank Adaptation) fine-tuning for domain adaptation.
    Uses 4-bit quantization with LoRA adapters.
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        self.is_initialized = False
        
    def initialize(self, base_model: str = None):
        """
        Initialize QLoRA with base model.
        
        Args:
            base_model: HuggingFace model ID or path
        """
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM, 
                AutoTokenizer,
                BitsAndBytesConfig
            )
            from peft import (
                LoraConfig,
                get_peft_model,
                prepare_model_for_kbit_training,
                PeftModel
            )
            
            base_model = base_model or "meta-llama/Llama-2-7b-hf"
            
            # Quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Prepare for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)
            
            # LoRA config
            lora_config = LoraConfig(
                r=self.config.qlora_r,
                lora_alpha=self.config.qlora_alpha,
                lora_dropout=self.config.qlora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            )
            
            # Apply LoRA
            self.peft_model = get_peft_model(self.model, lora_config)
            self.peft_model.print_trainable_parameters()
            
            self.is_initialized = True
            logger.info("QLoRA initialized successfully")
            
        except ImportError as e:
            logger.warning(f"QLoRA dependencies not installed: {e}")
            logger.info("Run: pip install transformers peft bitsandbytes accelerate")
            self.is_initialized = False
    
    def create_training_data(self, qa_pairs: List[Dict]) -> List[Dict]:
        """
        Create training data from Q&A pairs.
        
        Args:
            qa_pairs: List of {'question': str, 'answer': str, 'context': str}
            
        Returns:
            Formatted training data
        """
        training_data = []
        
        for qa in qa_pairs:
            prompt = f"""### Context:
{qa.get('context', '')}

### Question:
{qa['question']}

### Answer:
{qa['answer']}"""
            
            training_data.append({
                'text': prompt,
                'question': qa['question'],
                'answer': qa['answer']
            })
        
        return training_data
    
    def train(self, training_data: List[Dict], output_dir: str = None, epochs: int = 3):
        """
        Fine-tune the model with QLoRA.
        
        Args:
            training_data: List of training examples
            output_dir: Output directory for adapter
            epochs: Number of training epochs
        """
        if not self.is_initialized:
            logger.error("QLoRA not initialized. Call initialize() first.")
            return
        
        try:
            from transformers import TrainingArguments, Trainer
            from datasets import Dataset
            
            output_dir = output_dir or self.config.qlora_model_path
            
            # Create dataset
            dataset = Dataset.from_list(training_data)
            
            def tokenize_function(examples):
                return self.tokenizer(
                    examples['text'],
                    truncation=True,
                    max_length=512,
                    padding='max_length'
                )
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                warmup_steps=100,
                learning_rate=2e-4,
                fp16=True,
                logging_steps=10,
                save_steps=100,
                save_total_limit=2
            )
            
            # Trainer
            trainer = Trainer(
                model=self.peft_model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=self.tokenizer
            )
            
            # Train
            trainer.train()
            
            # Save adapter
            self.peft_model.save_pretrained(output_dir)
            logger.info(f"QLoRA adapter saved to: {output_dir}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
    
    def load_adapter(self, adapter_path: str = None):
        """Load a trained LoRA adapter."""
        if not self.model:
            logger.error("Base model not loaded")
            return
        
        try:
            from peft import PeftModel
            
            adapter_path = adapter_path or self.config.qlora_model_path
            self.peft_model = PeftModel.from_pretrained(self.model, adapter_path)
            logger.info(f"Adapter loaded from: {adapter_path}")
            
        except Exception as e:
            logger.error(f"Failed to load adapter: {e}")
    
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate text using the fine-tuned model."""
        if not self.peft_model or not self.tokenizer:
            logger.error("Model not initialized")
            return ""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.peft_model.device)
        
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# =============================================================================
# OLLAMA CLIENT
# =============================================================================

class OllamaClient:
    """
    Client for Ollama LLM API.
    Handles generation with retry logic.
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.base_url = config.ollama_url
        self.model = config.ollama_model
        self.timeout = config.ollama_timeout
    
    def generate(self, prompt: str, system_prompt: str = None, 
                temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """
        Generate response from Ollama.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        import requests
        
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '')
            
        except requests.exceptions.Timeout:
            logger.error(f"Ollama timeout after {self.timeout}s")
            return ""
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            return ""
    
    def chat(self, messages: List[Dict], temperature: float = 0.7, 
            max_tokens: int = 1024) -> str:
        """
        Chat completion with Ollama.
        
        Args:
            messages: List of {'role': str, 'content': str}
            temperature: Generation temperature
            max_tokens: Maximum tokens
            
        Returns:
            Generated response
        """
        import requests
        
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            return result.get('message', {}).get('content', '')
            
        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            return ""
    
    def embeddings(self, text: str) -> List[float]:
        """Get embeddings from Ollama."""
        import requests
        
        url = f"{self.base_url}/api/embeddings"
        
        payload = {
            "model": self.model,
            "prompt": text
        }
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            return result.get('embedding', [])
            
        except Exception as e:
            logger.error(f"Ollama embeddings failed: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        import requests
        
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


# =============================================================================
# GOLD STANDARD TRAINER
# =============================================================================

class GoldStandardTrainer:
    """
    Gold Standard pattern matching for high-quality responses.
    Matches queries to verified answer patterns.
    """
    
    def __init__(self):
        self.patterns: List[Dict] = []
        self._pattern_cache: Dict[str, Dict] = {}
    
    def add_pattern(self, pattern_id: str, trigger_phrases: List[str], 
                   concept: str, must_retrieve: Dict, answer_guidance: Dict):
        """Add a gold standard pattern."""
        pattern = {
            'id': pattern_id,
            'trigger_phrases': trigger_phrases,
            'concept': concept,
            'must_retrieve': must_retrieve,
            'answer_guidance': answer_guidance
        }
        self.patterns.append(pattern)
        
        # Index by trigger phrases
        for phrase in trigger_phrases:
            self._pattern_cache[phrase.lower()] = pattern
    
    def match_query_to_pattern(self, query: str) -> Optional[Dict]:
        """Match query to a gold standard pattern."""
        query_lower = query.lower()
        
        # Exact phrase match
        for phrase, pattern in self._pattern_cache.items():
            if phrase in query_lower:
                return pattern
        
        # Fuzzy match
        for pattern in self.patterns:
            for phrase in pattern['trigger_phrases']:
                phrase_words = set(phrase.lower().split())
                query_words = set(query_lower.split())
                
                if len(phrase_words & query_words) >= len(phrase_words) * 0.6:
                    return pattern
        
        return None
    
    def get_retrieval_targets(self, query: str) -> Dict[str, List[str]]:
        """Get required retrieval targets for query."""
        targets = {
            'sections': [],
            'tables': [],
            'figures': [],
            'appendices': []
        }
        
        pattern = self.match_query_to_pattern(query)
        if pattern:
            must_retrieve = pattern.get('must_retrieve', {})
            targets['sections'] = must_retrieve.get('sections', [])
            targets['tables'] = must_retrieve.get('tables', [])
            targets['figures'] = must_retrieve.get('figures', [])
            targets['appendices'] = must_retrieve.get('appendices', [])
        
        return targets
    
    def get_answer_guidance(self, query: str) -> Optional[Dict]:
        """Get answer structure guidance for query."""
        pattern = self.match_query_to_pattern(query)
        
        if pattern:
            guidance = pattern.get('answer_guidance', {})
            return {
                'concept': pattern.get('concept'),
                'pattern_id': pattern.get('id'),
                'start_with': guidance.get('start_with'),
                'must_mention': guidance.get('must_mention', []),
                'must_explain': guidance.get('must_explain', [])
            }
        return None
    
    def validate_answer(self, answer: str, query: str) -> Dict:
        """Validate answer against gold standard."""
        pattern = self.match_query_to_pattern(query)
        
        if not pattern:
            return {'valid': True, 'score': 0.5, 'missing': [], 'note': 'No matching pattern'}
        
        answer_lower = answer.lower()
        guidance = pattern.get('answer_guidance', {})
        must_mention = guidance.get('must_mention', [])
        
        mentioned = []
        missing = []
        
        for item in must_mention:
            if item.lower() in answer_lower:
                mentioned.append(item)
            else:
                missing.append(item)
        
        score = len(mentioned) / len(must_mention) if must_mention else 0.5
        
        return {
            'valid': score >= 0.5,
            'score': round(score, 2),
            'mentioned': mentioned,
            'missing': missing,
            'pattern_id': pattern['id']
        }


# =============================================================================
# CACHE MANAGER
# =============================================================================

class CacheManager:
    """Query response cache with TTL and LRU eviction."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.cache: Dict[str, Dict] = {}
        self.stats = {
            'hits': 0,
            'misses': 0,
            'total_queries': 0
        }
    
    def _normalize_key(self, query: str) -> str:
        """Normalize query for cache key."""
        import string
        query_clean = query.lower().translate(str.maketrans('', '', string.punctuation))
        words = query_clean.split()
        stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'does', 'do', 'can', 'how'}
        significant_words = [w for w in words if w not in stop_words and len(w) > 2]
        return ' '.join(sorted(significant_words))
    
    def get(self, query: str) -> Optional[Dict]:
        """Get cached response."""
        if not self.config.cache_enabled:
            return None
        
        cache_key = self._normalize_key(query)
        
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            age = time.time() - entry['timestamp']
            
            if age < self.config.cache_ttl_seconds:
                self.stats['hits'] += 1
                self.stats['total_queries'] += 1
                logger.debug(f"Cache HIT: {query[:50]}...")
                return entry
            else:
                del self.cache[cache_key]
        
        self.stats['misses'] += 1
        self.stats['total_queries'] += 1
        return None
    
    def set(self, query: str, answer: str, metadata: Dict = None):
        """Cache a response."""
        if not self.config.cache_enabled:
            return
        
        cache_key = self._normalize_key(query)
        
        # LRU eviction
        if len(self.cache) >= self.config.cache_max_size and cache_key not in self.cache:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        self.cache[cache_key] = {
            'original_query': query,
            'answer': answer,
            'metadata': metadata or {},
            'timestamp': time.time()
        }
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        hit_rate = (self.stats['hits'] / self.stats['total_queries'] * 100) if self.stats['total_queries'] > 0 else 0
        
        return {
            'enabled': self.config.cache_enabled,
            'total_queries': self.stats['total_queries'],
            'cache_hits': self.stats['hits'],
            'cache_misses': self.stats['misses'],
            'hit_rate_percent': round(hit_rate, 2),
            'current_size': len(self.cache),
            'max_size': self.config.cache_max_size
        }


# =============================================================================
# ENTITY EXTRACTOR
# =============================================================================

class EntityExtractor:
    """
    Extract entities from queries using pattern matching and NLP.
    """
    
    def __init__(self):
        self.patterns: Dict[str, List[str]] = {}
        self.acronym_map: Dict[str, str] = {}
    
    def add_pattern(self, entity_type: str, patterns: List[str]):
        """Add entity patterns."""
        if entity_type not in self.patterns:
            self.patterns[entity_type] = []
        self.patterns[entity_type].extend(patterns)
    
    def add_acronym(self, acronym: str, expansion: str):
        """Add acronym expansion."""
        self.acronym_map[acronym.upper()] = expansion
    
    def extract(self, text: str) -> List[Dict]:
        """
        Extract entities from text.
        
        Returns:
            List of {'entity': str, 'type': str, 'confidence': float}
        """
        entities = []
        text_upper = text.upper()
        
        # Pattern matching
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                if pattern.upper() in text_upper:
                    entities.append({
                        'entity': pattern,
                        'type': entity_type,
                        'confidence': 0.9
                    })
        
        # Acronym detection
        words = re.findall(r'\b[A-Z]{2,6}\b', text)
        for word in words:
            if word in self.acronym_map:
                entities.append({
                    'entity': word,
                    'type': 'acronym',
                    'expansion': self.acronym_map[word],
                    'confidence': 0.95
                })
        
        return entities
    
    def expand_acronyms(self, text: str) -> str:
        """Expand acronyms in text."""
        for acronym, expansion in self.acronym_map.items():
            text = re.sub(rf'\b{acronym}\b', f"{acronym} ({expansion})", text, count=1)
        return text


# =============================================================================
# RAG GRAPH QLORA SYSTEM
# =============================================================================

class RAGGraphQLoRA:
    """
    Main RAG Graph QLoRA System.
    Integrates all components for graph-based RAG with QLoRA fine-tuning.
    """
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        
        # Initialize components
        self.knowledge_graph = KnowledgeGraph()
        self.path_finder = NHopPathFinder(self.knowledge_graph)
        self.vector_db = VectorDatabase(self.config)
        self.reranker = HybridReranker(self.config)
        self.ollama = OllamaClient(self.config)
        self.qlora = QLoRAFineTuner(self.config) if self.config.qlora_enabled else None
        self.gold_trainer = GoldStandardTrainer()
        self.cache = CacheManager(self.config)
        self.entity_extractor = EntityExtractor()
        
        logger.info("RAG Graph QLoRA System initialized")
    
    def load_knowledge_graph(self, path: str):
        """Load knowledge graph from file."""
        self.knowledge_graph = KnowledgeGraph(json_path=path)
        self.path_finder = NHopPathFinder(self.knowledge_graph)
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to vector database."""
        self.vector_db.add_documents(documents)
    
    def add_gold_pattern(self, pattern_id: str, trigger_phrases: List[str],
                        concept: str, must_retrieve: Dict, answer_guidance: Dict):
        """Add a gold standard pattern."""
        self.gold_trainer.add_pattern(
            pattern_id, trigger_phrases, concept, must_retrieve, answer_guidance
        )
    
    def initialize_qlora(self, base_model: str = None):
        """Initialize QLoRA fine-tuning."""
        if self.qlora:
            self.qlora.initialize(base_model)
    
    def train_qlora(self, qa_pairs: List[Dict], epochs: int = 3):
        """Train QLoRA on Q&A pairs."""
        if self.qlora and self.qlora.is_initialized:
            training_data = self.qlora.create_training_data(qa_pairs)
            self.qlora.train(training_data, epochs=epochs)
    
    def query(self, question: str, use_cache: bool = True) -> Dict:
        """
        Process a query through the RAG pipeline.
        
        Args:
            question: User question
            use_cache: Whether to use cache
            
        Returns:
            Response dictionary with answer and metadata
        """
        start_time = time.time()
        
        # Check cache
        if use_cache:
            cached = self.cache.get(question)
            if cached:
                return {
                    'answer': cached['answer'],
                    'cached': True,
                    'metadata': cached['metadata']
                }
        
        # Extract entities
        entities = self.entity_extractor.extract(question)
        entity_names = [e['entity'] for e in entities]
        
        # Check gold standard
        gold_guidance = self.gold_trainer.get_answer_guidance(question)
        
        # Get graph context
        graph_context = self.path_finder.get_context_for_query(
            entity_names, question
        )
        
        # Vector search
        vector_results = self.vector_db.query(question)
        
        # Re-rank results
        reranked_results = self.reranker.rerank(question, vector_results)
        
        # Build context
        context_parts = []
        
        if graph_context['context_text']:
            context_parts.append("### GRAPH RELATIONSHIPS ###")
            context_parts.append(graph_context['context_text'])
        
        if reranked_results:
            context_parts.append("\n### RETRIEVED DOCUMENTS ###")
            for i, result in enumerate(reranked_results[:5], 1):
                context_parts.append(f"\n[Doc {i}]: {result['content'][:500]}...")
        
        context = "\n".join(context_parts)
        
        # Build prompt
        if gold_guidance:
            system_prompt = f"""You are a knowledgeable assistant. Answer based on the provided context.
Start your answer with: "{gold_guidance.get('start_with', '')}"
Must mention: {', '.join(gold_guidance.get('must_mention', []))}"""
        else:
            system_prompt = """You are a knowledgeable assistant. 
Answer questions accurately based on the provided context.
Cite sources when available. Be concise but thorough."""
        
        prompt = f"""### Context:
{context}

### Question:
{question}

### Answer:"""
        
        # Generate answer
        if self.qlora and self.qlora.is_initialized:
            answer = self.qlora.generate(prompt)
        else:
            answer = self.ollama.generate(prompt, system_prompt=system_prompt)
        
        # Validate against gold standard
        validation = self.gold_trainer.validate_answer(answer, question)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(answer, reranked_results, validation)
        
        # Extract citations
        citations = self._extract_citations(answer)
        
        # Build response
        response = {
            'answer': answer,
            'cached': False,
            'question': question,
            'entities': entities,
            'graph_paths': graph_context['paths'][:5],
            'retrieval_count': len(reranked_results),
            'citations': citations,
            'quality_score': quality_score,
            'gold_validation': validation,
            'timing': {
                'total_seconds': round(time.time() - start_time, 2)
            },
            'metadata': {
                'model': self.config.ollama_model,
                'used_qlora': self.qlora is not None and self.qlora.is_initialized,
                'graph_relationships': graph_context['relationship_count']
            }
        }
        
        # Cache response
        if use_cache:
            self.cache.set(question, answer, response)
        
        return response
    
    def _calculate_quality_score(self, answer: str, results: List[Dict], 
                                 validation: Dict) -> float:
        """Calculate answer quality score."""
        score = 0.0
        
        # Length score (prefer substantial answers)
        if len(answer) > 100:
            score += 0.2
        if len(answer) > 300:
            score += 0.1
        
        # Citation score
        citations = self._extract_citations(answer)
        if citations:
            score += min(0.3, len(citations) * 0.1)
        
        # Gold validation score
        if validation.get('valid'):
            score += validation.get('score', 0) * 0.4
        
        return min(1.0, score)
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract citations from text."""
        patterns = [
            r'C\d+\.\d+(?:\.\d+)*',  # C5.4.2.1
            r'Table\s+C?\d+\.T\d+[a-z]?',  # Table C5.T1
            r'Figure\s+C?\d+\.F\d+',  # Figure C5.F14
            r'Appendix\s+\d+',  # Appendix 6
            r'Section\s+\d+(?:\.\d+)*',  # Section 5.4
            r'Chapter\s+\d+'  # Chapter 5
        ]
        
        citations = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            citations.update(matches)
        
        return list(citations)
    
    def get_status(self) -> Dict:
        """Get system status."""
        return {
            'ollama_available': self.ollama.is_available(),
            'ollama_model': self.config.ollama_model,
            'vector_db_documents': self.vector_db.get_collection_count(),
            'knowledge_graph_entities': len(self.knowledge_graph.entities),
            'knowledge_graph_relationships': len(self.knowledge_graph.relationships),
            'gold_patterns': len(self.gold_trainer.patterns),
            'qlora_enabled': self.config.qlora_enabled,
            'qlora_initialized': self.qlora.is_initialized if self.qlora else False,
            'cache_stats': self.cache.get_stats()
        }


# =============================================================================
# FLASK API (OPTIONAL)
# =============================================================================

def create_api(rag_system: RAGGraphQLoRA):
    """Create Flask API for the RAG system."""
    try:
        from flask import Flask, request, jsonify
        from flask_cors import CORS
        
        app = Flask(__name__)
        CORS(app)
        
        @app.route('/api/query', methods=['POST'])
        def query():
            data = request.json
            question = data.get('question', '')
            
            if not question:
                return jsonify({'error': 'No question provided'}), 400
            
            result = rag_system.query(question)
            return jsonify(result)
        
        @app.route('/api/status', methods=['GET'])
        def status():
            return jsonify(rag_system.get_status())
        
        @app.route('/api/documents', methods=['POST'])
        def add_documents():
            data = request.json
            documents = data.get('documents', [])
            
            if not documents:
                return jsonify({'error': 'No documents provided'}), 400
            
            rag_system.add_documents(documents)
            return jsonify({'status': 'success', 'count': len(documents)})
        
        @app.route('/api/health', methods=['GET'])
        def health():
            return jsonify({'status': 'healthy', 'ollama': rag_system.ollama.is_available()})
        
        return app
        
    except ImportError:
        logger.warning("Flask not installed. Run: pip install flask flask-cors")
        return None


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def main():
    """Example usage of RAG Graph QLoRA system."""
    
    # Initialize configuration
    config = RAGConfig(
        ollama_url="http://localhost:11434",
        ollama_model="llama3.1:8b",
        vector_db_path="./chroma_db",
        qlora_enabled=False  # Set to True to enable QLoRA
    )
    
    # Create RAG system
    rag = RAGGraphQLoRA(config)
    
    # Add some sample entities to the knowledge graph
    rag.knowledge_graph.add_entity("entity1", {
        'id': 'entity1',
        'type': 'concept',
        'properties': {
            'label': 'Machine Learning',
            'definition': 'A subset of AI that enables systems to learn from data'
        }
    })
    
    rag.knowledge_graph.add_entity("entity2", {
        'id': 'entity2', 
        'type': 'concept',
        'properties': {
            'label': 'Deep Learning',
            'definition': 'A subset of ML using neural networks with many layers'
        }
    })
    
    rag.knowledge_graph.add_relationship(
        "entity2", "entity1", "subset_of",
        description="Deep Learning is a subset of Machine Learning"
    )
    
    # Add gold standard pattern
    rag.add_gold_pattern(
        pattern_id="ML_DEFINITION",
        trigger_phrases=["what is machine learning", "define ML", "ML definition"],
        concept="Machine Learning",
        must_retrieve={'sections': ['ML.1'], 'tables': [], 'figures': []},
        answer_guidance={
            'start_with': "Machine Learning is",
            'must_mention': ["AI", "data", "learning", "algorithms"],
            'must_explain': ["What ML is", "How it works", "Applications"]
        }
    )
    
    # Add sample documents
    sample_docs = [
        {
            'id': 'doc1',
            'content': 'Machine Learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.',
            'metadata': {'section': 'ML.1', 'type': 'definition'}
        },
        {
            'id': 'doc2',
            'content': 'Deep Learning uses neural networks with many layers to process complex patterns in data.',
            'metadata': {'section': 'DL.1', 'type': 'definition'}
        }
    ]
    rag.add_documents(sample_docs)
    
    # Check status
    print("\n=== System Status ===")
    status = rag.get_status()
    for key, value in status.items():
        print(f"{key}: {value}")
    
    # Query the system
    if rag.ollama.is_available():
        print("\n=== Query Test ===")
        question = "What is machine learning?"
        result = rag.query(question)
        
        print(f"Question: {question}")
        print(f"Answer: {result['answer'][:500]}...")
        print(f"Quality Score: {result['quality_score']}")
        print(f"Citations: {result['citations']}")
        print(f"Graph Paths: {len(result['graph_paths'])}")
        print(f"Timing: {result['timing']}")
    else:
        print("\nOllama not available. Start Ollama to test query functionality.")
    
    # Create API (optional)
    app = create_api(rag)
    if app:
        print("\n=== Starting API Server ===")
        print("API endpoints:")
        print("  POST /api/query - Query the RAG system")
        print("  GET  /api/status - Get system status")
        print("  POST /api/documents - Add documents")
        print("  GET  /api/health - Health check")
        # Uncomment to run: app.run(host='0.0.0.0', port=5000)


if __name__ == '__main__':
    main()

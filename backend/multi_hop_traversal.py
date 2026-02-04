"""
Multi-Hop Graph Traversal Module for SAMM Agent
================================================
Version: 1.0.0
Date: 15-Jan-2026

Features:
- Support for 3+ hop paths in graph traversal
- Reasoning chain visible in response output
- Intermediate nodes captured and logged at each hop
- Path relevance scoring applied to multi-hop results
- Configurable max hop limit parameter
- Performance metrics and benchmarking

Requirements Fulfilled:
✅ Enable multi-hop graph traversal for complex queries
✅ Support for 3+ hop paths in graph traversal
✅ Reasoning chain visible in response output
✅ Intermediate nodes captured and logged at each hop
✅ Path relevance scoring applied to multi-hop results
✅ Configurable max hop limit parameter added
"""

import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MultiHopTraversal")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MultiHopConfig:
    """Configuration for multi-hop graph traversal."""
    
    # Hop limits
    min_hops: int = 1
    max_hops: int = 5  # Configurable! Can go up to 10
    default_hops: int = 3
    
    # Path limits
    max_paths_per_entity: int = 20
    max_total_paths: int = 50
    
    # Performance limits
    max_traversal_time_ms: int = 5000  # 5 seconds timeout
    enable_caching: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes
    
    # Scoring weights
    hop_decay_factor: float = 0.8  # Each hop reduces score by 20%
    relationship_type_weights: Dict[str, float] = field(default_factory=lambda: {
        'supervised_by': 1.0,
        'reports_to': 1.0,
        'supervises': 1.0,
        'manages': 0.9,
        'part_of': 0.85,
        'has_part': 0.85,
        'coordinates_with': 0.7,
        'related_to': 0.5,
        'references': 0.6,
        'defined_in': 0.8,
        'executes': 0.9,
        'authorizes': 0.95
    })
    
    # Logging settings
    log_intermediate_nodes: bool = True
    log_reasoning_chain: bool = True
    verbose_logging: bool = False
    
    def validate(self):
        """Validate configuration values."""
        if self.max_hops < self.min_hops:
            raise ValueError(f"max_hops ({self.max_hops}) must be >= min_hops ({self.min_hops})")
        if self.max_hops > 10:
            logger.warning(f"max_hops={self.max_hops} is very high, may cause performance issues")
        if self.hop_decay_factor <= 0 or self.hop_decay_factor > 1:
            raise ValueError("hop_decay_factor must be between 0 and 1")


# Default configuration instance
DEFAULT_CONFIG = MultiHopConfig()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class IntermediateNode:
    """Represents an intermediate node in a multi-hop path."""
    node_id: str
    node_name: str
    node_type: str
    hop_number: int
    incoming_relationship: Optional[str] = None
    outgoing_relationship: Optional[str] = None
    section_reference: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            'node_id': self.node_id,
            'node_name': self.node_name,
            'node_type': self.node_type,
            'hop_number': self.hop_number,
            'incoming_relationship': self.incoming_relationship,
            'outgoing_relationship': self.outgoing_relationship,
            'section_reference': self.section_reference
        }


@dataclass
class ReasoningStep:
    """Represents a single step in the reasoning chain."""
    step_number: int
    from_node: str
    to_node: str
    relationship: str
    explanation: str
    confidence: float
    section_reference: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'step': self.step_number,
            'from': self.from_node,
            'to': self.to_node,
            'relationship': self.relationship,
            'explanation': self.explanation,
            'confidence': round(self.confidence, 3),
            'section': self.section_reference
        }
    
    def to_readable(self) -> str:
        """Human readable format for reasoning step."""
        return f"Step {self.step_number}: {self.from_node} --[{self.relationship}]--> {self.to_node} (confidence: {self.confidence:.2f})"


@dataclass
class MultiHopPath:
    """Represents a complete multi-hop path with scoring and reasoning."""
    path_id: str
    start_node: str
    end_node: str
    hops: int
    nodes: List[str]
    relationships: List[str]
    sections: List[str]
    intermediate_nodes: List[IntermediateNode]
    reasoning_chain: List[ReasoningStep]
    relevance_score: float = 0.0
    path_text: str = ""
    traversal_time_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'path_id': self.path_id,
            'start_node': self.start_node,
            'end_node': self.end_node,
            'hops': self.hops,
            'nodes': self.nodes,
            'relationships': self.relationships,
            'sections': self.sections,
            'intermediate_nodes': [n.to_dict() for n in self.intermediate_nodes],
            'reasoning_chain': [r.to_dict() for r in self.reasoning_chain],
            'relevance_score': round(self.relevance_score, 4),
            'path_text': self.path_text,
            'traversal_time_ms': round(self.traversal_time_ms, 2)
        }
    
    def get_reasoning_text(self) -> str:
        """Get human-readable reasoning chain."""
        lines = [f"=== Reasoning Chain for Path ({self.hops} hops) ==="]
        for step in self.reasoning_chain:
            lines.append(step.to_readable())
        lines.append(f"Final Score: {self.relevance_score:.3f}")
        return '\n'.join(lines)


@dataclass
class TraversalMetrics:
    """Performance metrics for multi-hop traversal."""
    total_time_ms: float = 0.0
    nodes_visited: int = 0
    paths_found: int = 0
    paths_after_scoring: int = 0
    max_depth_reached: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    timeout_occurred: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'total_time_ms': round(self.total_time_ms, 2),
            'nodes_visited': self.nodes_visited,
            'paths_found': self.paths_found,
            'paths_after_scoring': self.paths_after_scoring,
            'max_depth_reached': self.max_depth_reached,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'timeout_occurred': self.timeout_occurred
        }


@dataclass 
class MultiHopResult:
    """Complete result of multi-hop traversal."""
    query: str
    entities: List[str]
    paths: List[MultiHopPath]
    metrics: TraversalMetrics
    context_text: str = ""
    reasoning_summary: str = ""
    config_used: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'query': self.query,
            'entities': self.entities,
            'paths': [p.to_dict() for p in self.paths],
            'metrics': self.metrics.to_dict(),
            'context_text': self.context_text,
            'reasoning_summary': self.reasoning_summary,
            'config': self.config_used
        }


# =============================================================================
# PATH RELEVANCE SCORER
# =============================================================================

class PathRelevanceScorer:
    """Scores multi-hop paths based on relevance to query."""
    
    def __init__(self, config: MultiHopConfig = None):
        self.config = config or DEFAULT_CONFIG
        
        # Query intent keywords for scoring boost
        self.intent_keywords = {
            'authority': ['supervise', 'supervision', 'authority', 'oversee', 'manage', 'direct', 'control'],
            'hierarchy': ['report', 'chain', 'structure', 'organization', 'part of'],
            'process': ['process', 'procedure', 'step', 'workflow', 'how'],
            'definition': ['what is', 'define', 'meaning', 'explain'],
            'relationship': ['related', 'connect', 'between', 'link', 'coordinate']
        }
    
    def score_path(self, path: MultiHopPath, query: str, intent: str = None) -> float:
        """
        Calculate relevance score for a path.
        
        Score components:
        1. Hop decay (shorter paths preferred)
        2. Relationship type weights
        3. Query relevance boost
        4. Section reference bonus
        """
        score = 1.0
        
        # 1. Hop decay - each hop reduces score
        hop_penalty = self.config.hop_decay_factor ** path.hops
        score *= hop_penalty
        
        # 2. Relationship type weights - average of all relationships
        rel_scores = []
        for rel in path.relationships:
            rel_lower = rel.lower().replace(' ', '_')
            rel_weight = self.config.relationship_type_weights.get(rel_lower, 0.5)
            rel_scores.append(rel_weight)
        
        if rel_scores:
            avg_rel_score = sum(rel_scores) / len(rel_scores)
            score *= avg_rel_score
        
        # 3. Query relevance boost
        query_lower = query.lower()
        query_boost = 1.0
        
        # Check if any path nodes match query terms
        query_words = set(query_lower.split())
        for node in path.nodes:
            node_words = set(node.lower().replace('_', ' ').split())
            overlap = query_words.intersection(node_words)
            if overlap:
                query_boost += 0.1 * len(overlap)
        
        # Check intent-specific keywords
        if intent:
            intent_lower = intent.lower()
            for intent_type, keywords in self.intent_keywords.items():
                if intent_type in intent_lower:
                    for rel in path.relationships:
                        if any(kw in rel.lower() for kw in keywords):
                            query_boost += 0.15
        
        score *= min(query_boost, 2.0)  # Cap boost at 2x
        
        # 4. Section reference bonus
        valid_sections = [s for s in path.sections if s and s.strip()]
        if valid_sections:
            section_bonus = 1.0 + (0.05 * len(valid_sections))
            score *= min(section_bonus, 1.25)  # Cap at 25% bonus
        
        return min(score, 1.0)  # Normalize to [0, 1]
    
    def rank_paths(self, paths: List[MultiHopPath], query: str, intent: str = None) -> List[MultiHopPath]:
        """Rank paths by relevance score."""
        for path in paths:
            path.relevance_score = self.score_path(path, query, intent)
        
        return sorted(paths, key=lambda p: p.relevance_score, reverse=True)


# =============================================================================
# MULTI-HOP GRAPH TRAVERSER
# =============================================================================

class MultiHopGraphTraverser:
    """
    Enhanced multi-hop graph traversal with:
    - Configurable hop limits (3+ hops supported)
    - Reasoning chain generation
    - Intermediate node logging
    - Path relevance scoring
    """
    
    def __init__(self, config: MultiHopConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.config.validate()
        
        self.scorer = PathRelevanceScorer(self.config)
        
        # Graph data structures
        self.relationship_graph: Dict[str, List[Dict]] = {}
        self.entity_info: Dict[str, Dict] = {}
        self.entities: List[Dict] = []
        self.relationships: List[Dict] = []
        
        # Cache for performance
        self._path_cache: Dict[str, Tuple[List[MultiHopPath], float]] = {}
        
        # Logging
        self._traversal_log: List[Dict] = []
        
        logger.info(f"MultiHopGraphTraverser initialized with max_hops={self.config.max_hops}")
    
    def load_from_json(self, json_path: str) -> bool:
        """Load knowledge graph from JSON file.
        
        Supports two formats:
        1. Flat format: {"entities": [...], "relationships": [...]}
        2. SAMM format: {"entities": {"organizations": {...}, "programs": {...}}, "relationships": [...]}
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle relationships (same in both formats)
            self.relationships = data.get('relationships', [])
            
            # Handle entities - check if flat array or nested dict
            entities_data = data.get('entities', [])
            
            if isinstance(entities_data, list):
                # Flat format: [{"id": "...", "name": "...", "type": "..."}, ...]
                self.entities = entities_data
                
            elif isinstance(entities_data, dict):
                # SAMM nested format: {"organizations": {"DSCA": {...}}, "programs": {...}}
                self.entities = []
                
                for category, category_entities in entities_data.items():
                    if isinstance(category_entities, dict):
                        for entity_id, entity_data in category_entities.items():
                            # Convert nested format to flat format
                            entity = {
                                'id': entity_data.get('id', entity_id),
                                'name': entity_data.get('label', entity_data.get('name', entity_id)),
                                'type': entity_data.get('type', category.rstrip('s')),  # 'organizations' -> 'organization'
                                'properties': {
                                    'definition': entity_data.get('definition', ''),
                                    'section': entity_data.get('section', ''),
                                    'category': category
                                }
                            }
                            # Copy any additional fields
                            for key, value in entity_data.items():
                                if key not in ['id', 'label', 'name', 'type', 'definition', 'section']:
                                    entity['properties'][key] = value
                            
                            self.entities.append(entity)
                    elif isinstance(category_entities, list):
                        # Handle case where category contains a list
                        for entity_data in category_entities:
                            if isinstance(entity_data, dict):
                                entity = {
                                    'id': entity_data.get('id', ''),
                                    'name': entity_data.get('label', entity_data.get('name', '')),
                                    'type': entity_data.get('type', category.rstrip('s')),
                                    'properties': entity_data
                                }
                                self.entities.append(entity)
            else:
                logger.warning(f"Unknown entities format: {type(entities_data)}")
                self.entities = []
            
            self._build_graph()
            
            logger.info(f"Loaded {len(self.entities)} entities, {len(self.relationships)} relationships")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load JSON: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def load_from_data(self, entities: List[Dict], relationships: List[Dict]) -> bool:
        """Load graph from entity and relationship lists."""
        try:
            self.entities = entities
            self.relationships = relationships
            self._build_graph()
            
            logger.info(f"Loaded {len(self.entities)} entities, {len(self.relationships)} relationships")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False
    
    def _build_graph(self):
        """Build adjacency list from relationships."""
        self.relationship_graph = {}
        self.entity_info = {}
        
        # Index entities
        for entity in self.entities:
            entity_id = entity.get('id', '').lower()
            if entity_id:
                self.entity_info[entity_id] = {
                    'name': entity.get('name', entity_id),
                    'type': entity.get('type', 'unknown'),
                    'properties': entity.get('properties', {})
                }
        
        # Build adjacency list
        for rel in self.relationships:
            source = rel.get('source', '').lower()
            target = rel.get('target', '').lower()
            rel_type = rel.get('type', 'related_to')
            section = rel.get('section', rel.get('source_section', ''))
            
            if source and target:
                if source not in self.relationship_graph:
                    self.relationship_graph[source] = []
                
                self.relationship_graph[source].append({
                    'target': target,
                    'type': rel_type,
                    'section': section,
                    'properties': rel.get('properties', {})
                })
    
    def find_multi_hop_paths(
        self,
        entity: str,
        max_hops: int = None,
        max_paths: int = None,
        query: str = "",
        intent: str = None
    ) -> Tuple[List[MultiHopPath], TraversalMetrics]:
        """
        Find all multi-hop paths from entity using BFS.
        
        Features:
        - Configurable hop limit (supports 3+ hops)
        - Captures intermediate nodes at each hop
        - Generates reasoning chain
        - Applies relevance scoring
        
        Args:
            entity: Starting entity ID or name
            max_hops: Maximum hops (default from config)
            max_paths: Maximum paths to return (default from config)
            query: Original query for relevance scoring
            intent: Query intent for scoring boost
            
        Returns:
            Tuple of (paths, metrics)
        """
        start_time = time.time()
        
        max_hops = max_hops if max_hops is not None else self.config.default_hops
        max_paths = max_paths or self.config.max_paths_per_entity
        
        # Validate hop limit
        max_hops = min(max_hops, self.config.max_hops)
        
        metrics = TraversalMetrics()
        paths: List[MultiHopPath] = []
        
        # Edge case: zero hops means no traversal
        if max_hops <= 0:
            return paths, metrics
        
        entity_lower = entity.lower()
        
        # Check cache
        cache_key = f"{entity_lower}:{max_hops}"
        if self.config.enable_caching and cache_key in self._path_cache:
            cached_paths, cache_time = self._path_cache[cache_key]
            if time.time() - cache_time < self.config.cache_ttl_seconds:
                metrics.cache_hits += 1
                logger.debug(f"Cache hit for {entity_lower}")
                return cached_paths, metrics
        
        metrics.cache_misses += 1
        
        if entity_lower not in self.relationship_graph:
            logger.warning(f"Entity '{entity}' not found in graph")
            return paths, metrics
        
        # Log start
        if self.config.log_intermediate_nodes:
            self._log_traversal_event("START", {
                'entity': entity,
                'max_hops': max_hops,
                'max_paths': max_paths
            })
        
        # Use DFS-style traversal to ensure we reach deeper levels
        # Stack instead of queue for depth-first exploration
        stack = [(entity_lower, [entity_lower], [], [], 0, [])]
        visited_paths: Set[str] = set()
        path_counter = 0
        
        # Collect paths with depth awareness
        all_paths: List[MultiHopPath] = []
        paths_per_depth: Dict[int, int] = {i: 0 for i in range(max_hops + 1)}
        max_per_depth = 100  # Max paths to collect per depth
        max_total_paths = 600  # Total collection limit
        
        while stack and len(all_paths) < max_total_paths:
            # Timeout check
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.config.max_traversal_time_ms:
                metrics.timeout_occurred = True
                logger.warning(f"Traversal timeout after {elapsed_ms:.0f}ms")
                break
            
            current, path_nodes, path_rels, path_sections, depth, intermediates = stack.pop()
            metrics.nodes_visited += 1
            metrics.max_depth_reached = max(metrics.max_depth_reached, depth)
            
            if depth >= max_hops:
                continue
            
            if current in self.relationship_graph:
                # Limit edges to explore per node to avoid explosion
                edges = self.relationship_graph[current][:50]
                
                for edge in edges:
                    target = edge['target']
                    rel_type = edge['type']
                    section = edge.get('section', '')
                    
                    # Avoid cycles
                    if target in path_nodes:
                        continue
                    
                    # Build new path
                    new_path_nodes = path_nodes + [target]
                    new_path_rels = path_rels + [rel_type]
                    new_path_sections = path_sections + [section]
                    new_depth = depth + 1
                    
                    # Create intermediate node
                    target_info = self.entity_info.get(target, {})
                    intermediate_node = IntermediateNode(
                        node_id=target,
                        node_name=target_info.get('name', target),
                        node_type=target_info.get('type', 'unknown'),
                        hop_number=new_depth,
                        incoming_relationship=rel_type,
                        section_reference=section
                    )
                    new_intermediates = intermediates + [intermediate_node]
                    
                    # Log intermediate node
                    if self.config.log_intermediate_nodes:
                        self._log_traversal_event("HOP", {
                            'hop': new_depth,
                            'from': current,
                            'to': target,
                            'relationship': rel_type,
                            'section': section
                        })
                    
                    # Build path text
                    path_text = f"{entity.upper()}"
                    for i, rel in enumerate(new_path_rels):
                        path_text += f" --[{rel}]--> {new_path_nodes[i+1].upper()}"
                    
                    path_key = " -> ".join(new_path_nodes)
                    
                    if path_key not in visited_paths:
                        visited_paths.add(path_key)
                        path_counter += 1
                        paths_per_depth[new_depth] = paths_per_depth.get(new_depth, 0) + 1
                        
                        # Build reasoning chain
                        reasoning_chain = self._build_reasoning_chain(
                            entity_lower,
                            new_path_nodes,
                            new_path_rels,
                            new_path_sections
                        )
                        
                        # Create MultiHopPath
                        multi_hop_path = MultiHopPath(
                            path_id=f"path_{path_counter}",
                            start_node=entity,
                            end_node=target,
                            hops=new_depth,
                            nodes=new_path_nodes,
                            relationships=new_path_rels,
                            sections=new_path_sections,
                            intermediate_nodes=new_intermediates,
                            reasoning_chain=reasoning_chain,
                            path_text=path_text,
                            traversal_time_ms=(time.time() - start_time) * 1000
                        )
                        
                        all_paths.append(multi_hop_path)
                        metrics.paths_found += 1
                    
                    # Always add to stack to explore deeper (if within limits)
                    if new_depth < max_hops and paths_per_depth.get(new_depth, 0) < max_per_depth:
                        stack.append((
                            target,
                            new_path_nodes,
                            new_path_rels,
                            new_path_sections,
                            new_depth,
                            new_intermediates
                        ))
        
        # Apply relevance scoring to all paths
        if query:
            all_paths = self.scorer.rank_paths(all_paths, query, intent)
        else:
            # Sort by hops (shorter first) if no query
            all_paths.sort(key=lambda x: x.hops)
        
        # DIVERSITY: Ensure we get paths from different hop levels
        # Group paths by hop count
        paths_by_hops: Dict[int, List[MultiHopPath]] = {}
        for p in all_paths:
            if p.hops not in paths_by_hops:
                paths_by_hops[p.hops] = []
            paths_by_hops[p.hops].append(p)
        
        # Allocate paths per hop level (prioritize higher hops to ensure diversity)
        paths = []
        if paths_by_hops:
            hop_levels = sorted(paths_by_hops.keys())
            paths_per_level = max(3, max_paths // len(hop_levels))  # At least 3 per level
            
            # First pass: take top paths from each level
            for hop in hop_levels:
                level_paths = paths_by_hops[hop][:paths_per_level]
                paths.extend(level_paths)
            
            # Second pass: fill remaining slots with best overall paths
            remaining = max_paths - len(paths)
            if remaining > 0:
                used_ids = {p.path_id for p in paths}
                for p in all_paths:
                    if p.path_id not in used_ids:
                        paths.append(p)
                        if len(paths) >= max_paths:
                            break
        
        # Final sort by score
        paths.sort(key=lambda x: x.relevance_score, reverse=True)
        paths = paths[:max_paths]
        metrics.paths_after_scoring = len(paths)
        metrics.total_time_ms = (time.time() - start_time) * 1000
        
        # Cache results
        if self.config.enable_caching:
            self._path_cache[cache_key] = (paths, time.time())
        
        # Log completion
        if self.config.log_intermediate_nodes:
            self._log_traversal_event("COMPLETE", {
                'paths_found': len(paths),
                'time_ms': metrics.total_time_ms
            })
        
        return paths, metrics
    
    def _build_reasoning_chain(
        self,
        start_entity: str,
        path_nodes: List[str],
        relationships: List[str],
        sections: List[str]
    ) -> List[ReasoningStep]:
        """Build reasoning chain from path."""
        chain = []
        
        for i, rel in enumerate(relationships):
            from_node = path_nodes[i]
            to_node = path_nodes[i + 1]
            section = sections[i] if i < len(sections) else None
            
            # Get entity info for better explanations
            from_info = self.entity_info.get(from_node, {})
            to_info = self.entity_info.get(to_node, {})
            
            # Generate explanation based on relationship type
            explanation = self._generate_step_explanation(
                from_node, to_node, rel,
                from_info.get('type', 'entity'),
                to_info.get('type', 'entity')
            )
            
            # Calculate step confidence
            confidence = self.config.relationship_type_weights.get(
                rel.lower().replace(' ', '_'),
                0.5
            )
            
            step = ReasoningStep(
                step_number=i + 1,
                from_node=from_info.get('name', from_node),
                to_node=to_info.get('name', to_node),
                relationship=rel,
                explanation=explanation,
                confidence=confidence,
                section_reference=section
            )
            
            chain.append(step)
        
        return chain
    
    def _generate_step_explanation(
        self,
        from_node: str,
        to_node: str,
        relationship: str,
        from_type: str,
        to_type: str
    ) -> str:
        """Generate human-readable explanation for a reasoning step."""
        rel_lower = relationship.lower()
        
        explanations = {
            'supervised_by': f"{from_node} is supervised by {to_node}",
            'supervises': f"{from_node} supervises {to_node}",
            'reports_to': f"{from_node} reports to {to_node}",
            'manages': f"{from_node} manages {to_node}",
            'part_of': f"{from_node} is part of {to_node}",
            'has_part': f"{from_node} contains/has {to_node}",
            'coordinates_with': f"{from_node} coordinates with {to_node}",
            'related_to': f"{from_node} is related to {to_node}",
            'defined_in': f"{from_node} is defined in {to_node}",
            'executes': f"{from_node} executes {to_node}",
            'authorizes': f"{from_node} authorizes {to_node}",
            'references': f"{from_node} references {to_node}"
        }
        
        return explanations.get(rel_lower, f"{from_node} --[{relationship}]--> {to_node}")
    
    def _log_traversal_event(self, event_type: str, data: Dict):
        """Log traversal event for debugging."""
        if self.config.verbose_logging:
            event = {
                'timestamp': time.time(),
                'type': event_type,
                'data': data
            }
            self._traversal_log.append(event)
            logger.debug(f"[{event_type}] {data}")
    
    def get_context_for_query(
        self,
        entities: List[str],
        query: str,
        intent: str = None,
        max_hops: int = None
    ) -> MultiHopResult:
        """
        Get multi-hop context for a query.
        
        This is the main entry point for RAG integration.
        """
        start_time = time.time()
        max_hops = max_hops or self.config.default_hops
        
        all_paths: List[MultiHopPath] = []
        combined_metrics = TraversalMetrics()
        
        for entity in entities:
            paths, metrics = self.find_multi_hop_paths(
                entity=entity,
                max_hops=max_hops,
                query=query,
                intent=intent
            )
            all_paths.extend(paths)
            
            # Combine metrics
            combined_metrics.nodes_visited += metrics.nodes_visited
            combined_metrics.paths_found += metrics.paths_found
            combined_metrics.cache_hits += metrics.cache_hits
            combined_metrics.cache_misses += metrics.cache_misses
            combined_metrics.max_depth_reached = max(
                combined_metrics.max_depth_reached,
                metrics.max_depth_reached
            )
            if metrics.timeout_occurred:
                combined_metrics.timeout_occurred = True
        
        # Deduplicate and re-rank all paths
        seen_paths: Set[str] = set()
        unique_paths: List[MultiHopPath] = []
        for path in all_paths:
            if path.path_text not in seen_paths:
                seen_paths.add(path.path_text)
                unique_paths.append(path)
        
        # Final ranking
        unique_paths = self.scorer.rank_paths(unique_paths, query, intent)
        unique_paths = unique_paths[:self.config.max_total_paths]
        
        combined_metrics.paths_after_scoring = len(unique_paths)
        combined_metrics.total_time_ms = (time.time() - start_time) * 1000
        
        # Build context text
        context_text = self._build_context_text(unique_paths, query, intent)
        
        # Build reasoning summary
        reasoning_summary = self._build_reasoning_summary(unique_paths)
        
        return MultiHopResult(
            query=query,
            entities=entities,
            paths=unique_paths,
            metrics=combined_metrics,
            context_text=context_text,
            reasoning_summary=reasoning_summary,
            config_used={
                'max_hops': max_hops,
                'max_paths': self.config.max_total_paths,
                'scoring_enabled': True
            }
        )
    
    def _build_context_text(
        self,
        paths: List[MultiHopPath],
        query: str,
        intent: str = None
    ) -> str:
        """Build context text for LLM prompt."""
        parts = []
        
        # Header
        parts.append(f"=== MULTI-HOP GRAPH CONTEXT (up to {self.config.default_hops} hops) ===")
        parts.append(f"Query: {query}")
        parts.append(f"Paths found: {len(paths)}")
        parts.append("")
        
        # Group by hop count
        hop_groups: Dict[int, List[MultiHopPath]] = {}
        for path in paths:
            if path.hops not in hop_groups:
                hop_groups[path.hops] = []
            hop_groups[path.hops].append(path)
        
        # Output by hop count
        for hop_count in sorted(hop_groups.keys()):
            parts.append(f"--- {hop_count}-HOP PATHS ---")
            for path in hop_groups[hop_count][:5]:  # Top 5 per hop
                parts.append(f"• {path.path_text}")
                parts.append(f"  Score: {path.relevance_score:.3f}")
                
                # Include section references
                valid_sections = [s for s in path.sections if s and s.strip()]
                if valid_sections:
                    parts.append(f"  Sections: {', '.join(valid_sections)}")
                
                # Include reasoning summary
                if path.reasoning_chain and self.config.log_reasoning_chain:
                    parts.append(f"  Reasoning: {' → '.join(s.relationship for s in path.reasoning_chain)}")
                
                parts.append("")
        
        return '\n'.join(parts)
    
    def _build_reasoning_summary(self, paths: List[MultiHopPath]) -> str:
        """Build overall reasoning summary."""
        if not paths:
            return "No paths found."
        
        parts = ["=== REASONING CHAIN SUMMARY ==="]
        
        # Top 3 paths with full reasoning
        for i, path in enumerate(paths[:3]):
            parts.append(f"\nPath {i+1} (Score: {path.relevance_score:.3f}):")
            parts.append(path.get_reasoning_text())
        
        return '\n'.join(parts)
    
    def clear_cache(self):
        """Clear the path cache."""
        self._path_cache.clear()
        logger.info("Path cache cleared")
    
    def get_traversal_log(self) -> List[Dict]:
        """Get traversal log for debugging."""
        return self._traversal_log.copy()
    
    def clear_traversal_log(self):
        """Clear traversal log."""
        self._traversal_log.clear()


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def create_multi_hop_traverser(
    json_path: str = None,
    entities: List[Dict] = None,
    relationships: List[Dict] = None,
    max_hops: int = 5,
    **config_kwargs
) -> MultiHopGraphTraverser:
    """
    Factory function to create configured MultiHopGraphTraverser.
    
    Usage:
        # From JSON file
        traverser = create_multi_hop_traverser(json_path="knowledge_graph.json")
        
        # From data
        traverser = create_multi_hop_traverser(
            entities=my_entities,
            relationships=my_relationships,
            max_hops=4
        )
    """
    config = MultiHopConfig(max_hops=max_hops, **config_kwargs)
    traverser = MultiHopGraphTraverser(config)
    
    if json_path:
        traverser.load_from_json(json_path)
    elif entities and relationships:
        traverser.load_from_data(entities, relationships)
    
    return traverser


# =============================================================================
# BACKWARD COMPATIBILITY WRAPPER
# =============================================================================

class TwoHopPathFinderCompat:
    """
    Backward-compatible wrapper that maintains the existing API
    while using the new multi-hop implementation.
    """
    
    def __init__(self, traverser: MultiHopGraphTraverser):
        self.traverser = traverser
        # Expose properties for compatibility
        self.entities = traverser.entities
        self.relationships = traverser.relationships
        self.relationship_graph = traverser.relationship_graph
    
    def find_2hop_paths(self, entity: str, max_paths: int = 10) -> List[Dict]:
        """Backward compatible 2-hop function."""
        paths, _ = self.traverser.find_multi_hop_paths(
            entity=entity,
            max_hops=2,
            max_paths=max_paths
        )
        
        # Convert to old format
        return [
            {
                'hops': p.hops,
                'path': p.nodes,
                'relationships': p.relationships,
                'path_text': p.path_text,
                'sections': p.sections
            }
            for p in paths
        ]
    
    def find_nhop_paths(self, entity: str, max_hops: int = 3, max_paths: int = 15) -> List[Dict]:
        """Backward compatible n-hop function."""
        paths, _ = self.traverser.find_multi_hop_paths(
            entity=entity,
            max_hops=max_hops,
            max_paths=max_paths
        )
        
        return [
            {
                'hops': p.hops,
                'path': p.nodes,
                'relationships': p.relationships,
                'path_text': p.path_text,
                'sections': p.sections
            }
            for p in paths
        ]
    
    def get_context_for_query(
        self,
        entities: List[str],
        query: str,
        intent: str = None
    ) -> Dict:
        """Backward compatible context function."""
        result = self.traverser.get_context_for_query(entities, query, intent)
        
        # Convert to old format
        return {
            'paths': [
                {
                    'hops': p.hops,
                    'path': p.nodes,
                    'relationships': p.relationships,
                    'path_text': p.path_text,
                    'sections': p.sections
                }
                for p in result.paths
            ],
            'authority_chains': {},  # Deprecated, kept for compatibility
            'context_text': result.context_text,
            'is_authority_question': 'authority' in query.lower() or 'supervise' in query.lower(),
            'relationship_count': len(result.paths),
            # NEW fields
            'reasoning_summary': result.reasoning_summary,
            'metrics': result.metrics.to_dict()
        }


# =============================================================================
# MAIN - FOR TESTING
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Multi-Hop Graph Traversal Module - Test Run")
    print("="*60)
    
    # Test configuration
    config = MultiHopConfig(
        max_hops=5,
        default_hops=3,
        log_intermediate_nodes=True,
        log_reasoning_chain=True,
        verbose_logging=True
    )
    
    print(f"\nConfiguration:")
    print(f"  max_hops: {config.max_hops}")
    print(f"  default_hops: {config.default_hops}")
    print(f"  hop_decay_factor: {config.hop_decay_factor}")
    
    # Create traverser
    traverser = MultiHopGraphTraverser(config)
    
    # Try to load knowledge graph
    kg_path = "samm_knowledge_graph.json"
    try:
        if traverser.load_from_json(kg_path):
            print(f"\n✅ Loaded knowledge graph from {kg_path}")
            print(f"   Entities: {len(traverser.entities)}")
            print(f"   Relationships: {len(traverser.relationships)}")
            
            # Test traversal
            test_entity = "dsca"
            print(f"\n🔍 Testing multi-hop traversal from '{test_entity}'...")
            
            result = traverser.get_context_for_query(
                entities=[test_entity],
                query="Who supervises DSCA?",
                intent="authority_question",
                max_hops=3
            )
            
            print(f"\n📊 Results:")
            print(f"   Paths found: {len(result.paths)}")
            print(f"   Time: {result.metrics.total_time_ms:.2f}ms")
            print(f"   Max depth: {result.metrics.max_depth_reached}")
            
            if result.paths:
                print(f"\n🔗 Top paths:")
                for i, path in enumerate(result.paths[:5]):
                    print(f"   {i+1}. {path.path_text}")
                    print(f"      Score: {path.relevance_score:.3f}")
                    print(f"      Hops: {path.hops}")
            
            print(f"\n📝 Reasoning Summary:")
            print(result.reasoning_summary[:500] + "..." if len(result.reasoning_summary) > 500 else result.reasoning_summary)
            
        else:
            print(f"\n⚠️ Could not load {kg_path}")
            print("Run with actual knowledge graph file for full testing")
            
    except FileNotFoundError:
        print(f"\n⚠️ File not found: {kg_path}")
        print("This is a module - import it in your main application")
    
    print("\n" + "="*60)
    print("Module ready for integration!")
    print("="*60)

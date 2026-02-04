# =============================================================================
# REINFORCEMENT LEARNING & SELF-LEARNING SYSTEM
# Add these classes AFTER your existing DatabaseManager class
# =============================================================================

import numpy as np
from collections import deque, defaultdict
from datetime import datetime, timedelta
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import threading
import time

# =============================================================================
# 1. REINFORCEMENT LEARNING CORE
# =============================================================================

class ReinforcementLearningEngine:
    """
    Core RL engine using Q-learning for agent optimization
    Learns optimal strategies from user feedback
    """
    
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.learning_rate = learning_rate  # Alpha
        self.discount_factor = discount_factor  # Gamma
        self.epsilon = epsilon  # Exploration rate
        
        # Q-table: {state: {action: q_value}}
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=10000)
        
        # Performance tracking
        self.episode_rewards = []
        self.learning_stats = {
            "total_episodes": 0,
            "total_updates": 0,
            "average_reward": 0.0,
            "best_reward": float('-inf'),
            "convergence_rate": 0.0
        }
        
        # State-action frequency for exploration bonus
        self.state_action_counts = defaultdict(lambda: defaultdict(int))
        
        print("[RL Engine] Initialized with Q-learning")
    
    def get_state_representation(self, query: str, intent: str, entities: List[str],
                                 context: Dict) -> str:
        """Convert query context into discrete state representation"""
        # Create hash-able state representation
        state_features = [
            f"intent:{intent}",
            f"entities:{len(entities)}",
            f"query_length:{len(query.split())}",
            f"has_context:{bool(context)}"
        ]
        
        # Add entity types
        if entities:
            entity_types = set()
            for entity in entities[:3]:  # Top 3 entities
                if any(org in entity.upper() for org in ['DSCA', 'DFAS', 'DOD', 'DOS']):
                    entity_types.add("organization")
                elif any(prog in entity.upper() for prog in ['FMS', 'SA', 'SC']):
                    entity_types.add("program")
            state_features.extend([f"entity_type:{t}" for t in entity_types])
        
        return "|".join(sorted(state_features))
    
    def get_action_space(self) -> List[str]:
        """Define available actions for agents"""
        return [
            "standard_processing",
            "enhanced_extraction",
            "deep_database_query",
            "multi_pass_generation",
            "confidence_boosting",
            "context_expansion",
            "relationship_inference"
        ]
    
    def select_action(self, state: str, training: bool = True) -> str:
        """Select action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            action = np.random.choice(self.get_action_space())
            print(f"[RL] Exploring: {action}")
            return action
        
        # Exploitation: best known action
        q_values = self.q_table[state]
        if not q_values:
            action = np.random.choice(self.get_action_space())
            print(f"[RL] Random (no history): {action}")
            return action
        
        # Add exploration bonus (Upper Confidence Bound)
        adjusted_values = {}
        total_visits = sum(self.state_action_counts[state].values())
        
        for action in self.get_action_space():
            q_value = q_values.get(action, 0.0)
            visits = self.state_action_counts[state][action]
            
            # UCB bonus: encourages less-visited actions
            if total_visits > 0 and visits > 0:
                exploration_bonus = np.sqrt(2 * np.log(total_visits) / visits)
                adjusted_values[action] = q_value + 0.1 * exploration_bonus
            else:
                adjusted_values[action] = q_value
        
        action = max(adjusted_values.items(), key=lambda x: x[1])[0]
        print(f"[RL] Exploiting: {action} (Q={q_values.get(action, 0.0):.3f})")
        return action
    
    def calculate_reward(self, feedback_data: Dict) -> float:
        """
        Calculate reward from user feedback
        
        Feedback types:
        - approval: +10
        - rejection: -10
        - needs_revision: -5
        - confidence_boost: +5
        - accuracy_score: 0-10
        """
        reward = 0.0
        
        # Base feedback
        if feedback_data.get('approved', False):
            reward += 10.0
        elif feedback_data.get('rejected', False):
            reward -= 10.0
        elif feedback_data.get('needs_revision', False):
            reward -= 5.0
        
        # Accuracy scoring
        accuracy = feedback_data.get('accuracy_score', 0)  # 0-10 scale
        reward += accuracy
        
        # Response quality
        if feedback_data.get('complete_answer', False):
            reward += 3.0
        if feedback_data.get('correct_entities', False):
            reward += 2.0
        if feedback_data.get('correct_intent', False):
            reward += 2.0
        
        # Penalties
        if feedback_data.get('hallucination', False):
            reward -= 15.0
        if feedback_data.get('irrelevant', False):
            reward -= 8.0
        
        # Time efficiency bonus
        execution_time = feedback_data.get('execution_time', 0)
        if execution_time < 10:  # Fast response
            reward += 2.0
        elif execution_time > 30:  # Slow response
            reward -= 2.0
        
        print(f"[RL] Calculated reward: {reward:.2f}")
        return reward
    
    def update_q_value(self, state: str, action: str, reward: float, 
                       next_state: str, done: bool = False):
        """Update Q-value using Q-learning update rule"""
        current_q = self.q_table[state][action]
        
        if done:
            # Terminal state
            max_next_q = 0.0
        else:
            # Max Q-value for next state
            next_q_values = self.q_table[next_state]
            max_next_q = max(next_q_values.values()) if next_q_values else 0.0
        
        # Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
        self.state_action_counts[state][action] += 1
        
        self.learning_stats['total_updates'] += 1
        
        print(f"[RL] Q-update: {current_q:.3f} → {new_q:.3f} (reward={reward:.2f})")
    
    def store_experience(self, state: str, action: str, reward: float,
                        next_state: str, done: bool):
        """Store experience for replay"""
        self.experience_buffer.append((state, action, reward, next_state, done))
    
    def replay_experiences(self, batch_size: int = 32):
        """Experience replay for stable learning"""
        if len(self.experience_buffer) < batch_size:
            return
        
        # Sample random batch
        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in indices]
        
        # Update Q-values from experiences
        for state, action, reward, next_state, done in batch:
            self.update_q_value(state, action, reward, next_state, done)
        
        print(f"[RL] Replayed {batch_size} experiences")
    
    def save_model(self, filepath: str = "models/rl_model.pkl"):
        """Save Q-table and statistics"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'q_table': dict(self.q_table),
            'learning_stats': self.learning_stats,
            'state_action_counts': dict(self.state_action_counts),
            'experience_buffer': list(self.experience_buffer)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"[RL] Model saved to {filepath}")
    
    def load_model(self, filepath: str = "models/rl_model.pkl"):
        """Load Q-table and statistics"""
        if not Path(filepath).exists():
            print(f"[RL] No saved model found at {filepath}")
            return False
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = defaultdict(lambda: defaultdict(float), model_data['q_table'])
        self.learning_stats = model_data['learning_stats']
        self.state_action_counts = defaultdict(lambda: defaultdict(int), 
                                               model_data['state_action_counts'])
        self.experience_buffer = deque(model_data['experience_buffer'], maxlen=10000)
        
        print(f"[RL] Model loaded from {filepath}")
        return True
    
    def get_learning_stats(self) -> Dict:
        """Get learning statistics"""
        avg_q_value = 0.0
        total_states = len(self.q_table)
        
        if total_states > 0:
            all_q_values = []
            for state_actions in self.q_table.values():
                all_q_values.extend(state_actions.values())
            avg_q_value = np.mean(all_q_values) if all_q_values else 0.0
        
        return {
            **self.learning_stats,
            'total_states': total_states,
            'total_experiences': len(self.experience_buffer),
            'average_q_value': avg_q_value,
            'exploration_rate': self.epsilon
        }


# =============================================================================
# 2. SELF-LEARNING VECTOR DATABASE
# =============================================================================

class SelfLearningVectorDB:
    """
    Self-learning wrapper for ChromaDB
    Automatically optimizes embeddings and improves retrieval
    """
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        
        # Query-result feedback tracking
        self.query_feedback = []  # List of (query, results, feedback_score)
        
        # Performance metrics
        self.retrieval_metrics = {
            'total_queries': 0,
            'average_relevance': 0.0,
            'improvement_rate': 0.0
        }
        
        # Learned query patterns
        self.successful_patterns = defaultdict(list)  # intent → successful queries
        self.failed_patterns = defaultdict(list)  # intent → failed queries
        
        # Embedding optimization
        self.embedding_adjustments = {}  # entity → adjustment_vector
        
        # Auto-retraining flag
        self.needs_retraining = False
        self.last_retrain_time = datetime.now()
        
        print("[Self-Learning Vector DB] Initialized")
    
    def query_with_learning(self, query: str, intent: str, n_results: int = 10) -> List[Dict]:
        """Query with learning from past feedback"""
        self.retrieval_metrics['total_queries'] += 1
        
        # Check if we have successful patterns for this intent
        if intent in self.successful_patterns:
            # Apply learned query optimization
            query = self._optimize_query(query, intent)
        
        # Get results from vector DB
        results = self.db_manager.query_vector_db(query, n_results=n_results)
        
        # Apply learned reranking
        results = self._rerank_results(results, query, intent)
        
        return results
    
    def _optimize_query(self, query: str, intent: str) -> str:
        """Optimize query based on successful patterns"""
        successful = self.successful_patterns[intent]
        
        if not successful:
            return query
        
        # Find common terms in successful queries
        common_terms = self._extract_common_terms(successful)
        
        # Add high-value terms not in current query
        query_terms = set(query.lower().split())
        missing_terms = [term for term in common_terms if term not in query_terms]
        
        if missing_terms:
            optimized = f"{query} {' '.join(missing_terms[:2])}"
            print(f"[Self-Learning] Optimized query with: {missing_terms[:2]}")
            return optimized
        
        return query
    
    def _extract_common_terms(self, queries: List[str]) -> List[str]:
        """Extract common high-value terms from successful queries"""
        from collections import Counter
        
        # Count term frequencies
        term_counts = Counter()
        for query in queries:
            terms = query.lower().split()
            term_counts.update(terms)
        
        # Return most common terms (excluding stop words)
        stop_words = {'the', 'a', 'an', 'is', 'are', 'what', 'how', 'who', 'when', 'where'}
        common = [term for term, count in term_counts.most_common(10) 
                 if term not in stop_words and len(term) > 2]
        
        return common
    
    def _rerank_results(self, results: List[Dict], query: str, intent: str) -> List[Dict]:
        """Rerank results based on learned preferences"""
        if not results:
            return results
        
        # Calculate learned relevance scores
        for result in results:
            base_score = 1 - result.get('similarity', 0.5)  # Lower distance = higher score
            
            # Boost based on successful patterns
            boost = 0.0
            content = result.get('content', '').lower()
            
            if intent in self.successful_patterns:
                # Check if result matches successful patterns
                for successful_query in self.successful_patterns[intent][-5:]:
                    query_terms = set(successful_query.lower().split())
                    content_terms = set(content.split())
                    
                    overlap = len(query_terms & content_terms) / max(len(query_terms), 1)
                    boost += overlap * 0.1
            
            result['learned_score'] = base_score + boost
        
        # Sort by learned score
        results.sort(key=lambda x: x.get('learned_score', 0), reverse=True)
        
        return results
    
    def record_feedback(self, query: str, intent: str, results: List[Dict],
                       feedback_score: float, was_helpful: bool):
        """
        Record feedback for self-learning
        
        Args:
            query: Original query
            intent: Query intent
            results: Retrieved results
            feedback_score: 0.0-1.0 relevance score
            was_helpful: Boolean user satisfaction
        """
        self.query_feedback.append({
            'query': query,
            'intent': intent,
            'results': results,
            'feedback_score': feedback_score,
            'was_helpful': was_helpful,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update patterns
        if was_helpful and feedback_score > 0.7:
            self.successful_patterns[intent].append(query)
            print(f"[Self-Learning] Added successful pattern for {intent}")
        elif not was_helpful or feedback_score < 0.3:
            self.failed_patterns[intent].append(query)
            print(f"[Self-Learning] Added failed pattern for {intent}")
        
        # Update metrics
        current_avg = self.retrieval_metrics['average_relevance']
        total = self.retrieval_metrics['total_queries']
        new_avg = (current_avg * (total - 1) + feedback_score) / total
        self.retrieval_metrics['average_relevance'] = new_avg
        
        # Check if retraining needed
        if len(self.query_feedback) % 100 == 0:
            self.needs_retraining = True
            print(f"[Self-Learning] Retraining recommended ({len(self.query_feedback)} feedback entries)")
    
    def auto_retrain(self):
        """Automatically retrain embeddings based on feedback"""
        if not self.needs_retraining:
            return False
        
        if datetime.now() - self.last_retrain_time < timedelta(hours=1):
            print("[Self-Learning] Skipping retrain (too soon)")
            return False
        
        print("[Self-Learning] Starting auto-retrain...")
        
        # Analyze feedback to improve embeddings
        positive_feedback = [f for f in self.query_feedback if f['was_helpful']]
        negative_feedback = [f for f in self.query_feedback if not f['was_helpful']]
        
        print(f"[Self-Learning] Analyzing {len(positive_feedback)} positive, {len(negative_feedback)} negative examples")
        
        # Extract patterns
        improvements = self._calculate_embedding_improvements(positive_feedback, negative_feedback)
        
        if improvements:
            print(f"[Self-Learning] Generated {len(improvements)} embedding improvements")
            self.embedding_adjustments.update(improvements)
        
        self.needs_retraining = False
        self.last_retrain_time = datetime.now()
        
        return True
    
    def _calculate_embedding_improvements(self, positive_feedback: List[Dict],
                                         negative_feedback: List[Dict]) -> Dict:
        """Calculate embedding adjustments based on feedback"""
        improvements = {}
        
        # For each entity that appears in positive feedback,
        # boost its importance in the embedding space
        for feedback in positive_feedback:
            query = feedback['query']
            # Extract entities (simplified)
            entities = self._extract_entities_simple(query)
            
            for entity in entities:
                if entity not in improvements:
                    improvements[entity] = {'boost': 0.0, 'count': 0}
                
                improvements[entity]['boost'] += 0.1
                improvements[entity]['count'] += 1
        
        # Penalize entities in negative feedback
        for feedback in negative_feedback:
            query = feedback['query']
            entities = self._extract_entities_simple(query)
            
            for entity in entities:
                if entity not in improvements:
                    improvements[entity] = {'boost': 0.0, 'count': 0}
                
                improvements[entity]['boost'] -= 0.05
                improvements[entity]['count'] += 1
        
        return improvements
    
    def _extract_entities_simple(self, text: str) -> List[str]:
        """Simple entity extraction for feedback analysis"""
        # Capitalize words that might be entities
        words = text.split()
        entities = []
        
        for word in words:
            if len(word) > 2 and (word.isupper() or word[0].isupper()):
                entities.append(word)
        
        return entities
    
    def get_performance_metrics(self) -> Dict:
        """Get self-learning performance metrics"""
        return {
            **self.retrieval_metrics,
            'total_feedback_entries': len(self.query_feedback),
            'successful_patterns_count': sum(len(v) for v in self.successful_patterns.values()),
            'failed_patterns_count': sum(len(v) for v in self.failed_patterns.values()),
            'embedding_adjustments': len(self.embedding_adjustments),
            'needs_retraining': self.needs_retraining,
            'last_retrain': self.last_retrain_time.isoformat()
        }
    
    def save_learning_data(self, filepath: str = "models/vector_learning.json"):
        """Save learning data"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'query_feedback': self.query_feedback[-1000:],  # Last 1000
            'successful_patterns': dict(self.successful_patterns),
            'failed_patterns': dict(self.failed_patterns),
            'embedding_adjustments': self.embedding_adjustments,
            'metrics': self.retrieval_metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[Self-Learning] Saved learning data to {filepath}")
    
    def load_learning_data(self, filepath: str = "models/vector_learning.json"):
        """Load learning data"""
        if not Path(filepath).exists():
            print(f"[Self-Learning] No saved data at {filepath}")
            return False
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.query_feedback = data['query_feedback']
        self.successful_patterns = defaultdict(list, data['successful_patterns'])
        self.failed_patterns = defaultdict(list, data['failed_patterns'])
        self.embedding_adjustments = data['embedding_adjustments']
        self.retrieval_metrics = data['metrics']
        
        print(f"[Self-Learning] Loaded learning data from {filepath}")
        return True


# =============================================================================
# 3. SELF-LEARNING KNOWLEDGE GRAPH
# =============================================================================

class SelfLearningKnowledgeGraph:
    """
    Self-learning knowledge graph that automatically:
    - Discovers new entities and relationships
    - Infers missing relationships
    - Consolidates duplicate entities
    - Scores relationship confidence
    """
    
    def __init__(self, knowledge_graph, db_manager):
        self.kg = knowledge_graph
        self.db_manager = db_manager
        
        # Discovered knowledge
        self.discovered_entities = {}  # entity_id → entity_data
        self.discovered_relationships = []  # List of relationships
        
        # Confidence scores
        self.entity_confidence = {}  # entity_id → confidence (0-1)
        self.relationship_confidence = {}  # (source, rel, target) → confidence
        
        # Co-occurrence tracking
        self.entity_cooccurrence = defaultdict(lambda: defaultdict(int))
        
        # Relationship inference rules
        self.inference_rules = [
            # If A supervises B and B manages C, then A oversees C
            {
                'if': [('?A', 'supervises', '?B'), ('?B', 'manages', '?C')],
                'then': ('?A', 'oversees', '?C'),
                'confidence': 0.8
            },
            # If A is subset of B, then B includes A
            {
                'if': [('?A', 'isSubsetOf', '?B')],
                'then': ('?B', 'includes', '?A'),
                'confidence': 0.9
            }
        ]
        
        # Learning statistics
        self.learning_stats = {
            'entities_discovered': 0,
            'relationships_inferred': 0,
            'duplicates_merged': 0,
            'confidence_updates': 0
        }
        
        print("[Self-Learning KG] Initialized")
    
    def extract_from_query_context(self, query: str, answer: str, 
                                   retrieved_docs: List[Dict]):
        """Extract new entities and relationships from query/answer context"""
        print(f"[Self-Learning KG] Extracting from context...")
        
        # Extract entities from answer
        new_entities = self._extract_entities_from_text(answer)
        
        # Extract relationships from answer
        new_relationships = self._extract_relationships_from_text(answer)
        
        # Update co-occurrence from retrieved docs
        for doc in retrieved_docs:
            content = doc.get('content', '')
            doc_entities = self._extract_entities_from_text(content)
            self._update_cooccurrence(doc_entities)
        
        # Add discovered entities
        for entity in new_entities:
            if entity not in self.discovered_entities:
                self.discovered_entities[entity] = {
                    'discovered_from': 'query_context',
                    'timestamp': datetime.now().isoformat(),
                    'occurrences': 1
                }
                self.entity_confidence[entity] = 0.5  # Initial confidence
                self.learning_stats['entities_discovered'] += 1
            else:
                self.discovered_entities[entity]['occurrences'] += 1
                # Increase confidence with more occurrences
                self.entity_confidence[entity] = min(0.95, 
                    self.entity_confidence[entity] + 0.05)
        
        # Add discovered relationships
        for rel in new_relationships:
            if rel not in self.discovered_relationships:
                self.discovered_relationships.append(rel)
                rel_tuple = (rel['source'], rel['relationship'], rel['target'])
                self.relationship_confidence[rel_tuple] = 0.5
                self.learning_stats['relationships_inferred'] += 1
        
        print(f"[Self-Learning KG] Discovered {len(new_entities)} entities, {len(new_relationships)} relationships")
    
    def _extract_entities_from_text(self, text: str) -> List[str]:
        """Extract potential entities from text"""
        entities = []
        
        # Known SAMM entities
        samm_entities = ['DSCA', 'DFAS', 'DoD', 'DoS', 'AECA', 'FAA', 'NDAA',
                        'Security Cooperation', 'Security Assistance', 'FMS']
        
        for entity in samm_entities:
            if entity in text:
                entities.append(entity)
        
        # Capitalized phrases (potential entities)
        import re
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(capitalized[:5])  # Limit to avoid noise
        
        return list(set(entities))
    
    def _extract_relationships_from_text(self, text: str) -> List[Dict]:
        """Extract potential relationships from text using patterns"""
        relationships = []
        
        # Relationship patterns
        patterns = [
            (r'(\w+)\s+supervises\s+(\w+)', 'supervises'),
            (r'(\w+)\s+manages\s+(\w+)', 'manages'),
            (r'(\w+)\s+is responsible for\s+(\w+)', 'responsible_for'),
            (r'(\w+)\s+includes\s+(\w+)', 'includes'),
            (r'(\w+)\s+is part of\s+(\w+)', 'part_of'),
        ]
        
        import re
        for pattern, rel_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                relationships.append({
                    'source': match.group(1),
                    'relationship': rel_type,
                    'target': match.group(2)
                })
        
        return relationships
    
    def _update_cooccurrence(self, entities: List[str]):
        """Update entity co-occurrence matrix"""
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                self.entity_cooccurrence[entity1][entity2] += 1
                self.entity_cooccurrence[entity2][entity1] += 1
    
    def infer_missing_relationships(self):
        """Infer missing relationships using rules and co-occurrence"""
        print("[Self-Learning KG] Inferring missing relationships...")
        
        inferred = []
        
        # Rule-based inference
        for rule in self.inference_rules:
            inferred_rels = self._apply_inference_rule(rule)
            inferred.extend(inferred_rels)
        
        # Co-occurrence-based inference
        cooccurrence_rels = self._infer_from_cooccurrence()
        inferred.extend(cooccurrence_rels)
        
        # Add inferred relationships
        for rel in inferred:
            rel_tuple = (rel['source'], rel['relationship'], rel['target'])
            if rel_tuple not in self.relationship_confidence:
                self.relationship_confidence[rel_tuple] = rel['confidence']
                self.discovered_relationships.append(rel)
                self.learning_stats['relationships_inferred'] += 1
        
        print(f"[Self-Learning KG] Inferred {len(inferred)} new relationships")
        return inferred
    
    def _apply_inference_rule(self, rule: Dict) -> List[Dict]:
        """Apply a single inference rule"""
        inferred = []
        
        # For each combination of entities that match the rule conditions
        # (Simplified - in production, use proper graph query)
        
        # Example: If A supervises B and B manages C, then A oversees C
        if_patterns = rule['if']
        then_pattern = rule['then']
        confidence = rule['confidence']
        
        # Check existing relationships
        matching_entities = self._find_matching_patterns(if_patterns)
        
        for match in matching_entities:
            source = match.get('?A')
            target = match.get('?C')
            
            if source and target:
                inferred.append({
                    'source': source,
                    'relationship': then_pattern[1],
                    'target': target,
                    'confidence': confidence,
                    'inferred_by': 'rule'
                })
        
        return inferred
    
    def _find_matching_patterns(self, patterns: List[Tuple]) -> List[Dict]:
        """Find entity combinations matching rule patterns"""
        # Simplified pattern matching
        # In production, use proper graph traversal
        matches = []
        
        # Check discovered relationships
        for rel in self.discovered_relationships:
            # Match against patterns (simplified)
            # ... pattern matching logic ...
            pass
        
        return matches
    
    def _infer_from_cooccurrence(self, threshold: int = 5) -> List[Dict]:
        """Infer relationships from entity co-occurrence"""
        inferred = []
        
        for entity1, cooccurrences in self.entity_cooccurrence.items():
            for entity2, count in cooccurrences.items():
                if count >= threshold:
                    # High co-occurrence suggests relationship
                    confidence = min(0.7, count / 10)  # Scale confidence
                    
                    inferred.append({
                        'source': entity1,
                        'relationship': 'related_to',
                        'target': entity2,
                        'confidence': confidence,
                        'inferred_by': 'cooccurrence',
                        'evidence_count': count
                    })
        
        return inferred
    
    def consolidate_duplicate_entities(self):
        """Find and merge duplicate entities"""
        print("[Self-Learning KG] Consolidating duplicates...")
        
        entities_list = list(self.discovered_entities.keys())
        merged = 0
        
        for i, entity1 in enumerate(entities_list):
            for entity2 in entities_list[i+1:]:
                if self._are_duplicates(entity1, entity2):
                    # Merge entities
                    self._merge_entities(entity1, entity2)
                    merged += 1
        
        self.learning_stats['duplicates_merged'] = merged
        print(f"[Self-Learning KG] Merged {merged} duplicate entities")
    
    def _are_duplicates(self, entity1: str, entity2: str) -> bool:
        """Check if two entities are duplicates"""
        # Exact match (case-insensitive)
        if entity1.lower() == entity2.lower():
            return True
        
        # Acronym match
        if entity1.upper() in entity2 or entity2.upper() in entity1:
            return True
        
        # Levenshtein distance (simplified)
        if self._string_similarity(entity1, entity2) > 0.9:
            return True
        
        return False
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity (simplified Levenshtein)"""
        if len(s1) == 0 or len(s2) == 0:
            return 0.0
        
        # Simple character overlap
        set1 = set(s1.lower())
        set2 = set(s2.lower())
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_entities(self, entity1: str, entity2: str):
        """Merge two duplicate entities"""
        # Keep entity with higher confidence
        conf1 = self.entity_confidence.get(entity1, 0)
        conf2 = self.entity_confidence.get(entity2, 0)
        
        primary = entity1 if conf1 >= conf2 else entity2
        secondary = entity2 if primary == entity1 else entity1
        
        # Merge data
        if primary in self.discovered_entities and secondary in self.discovered_entities:
            self.discovered_entities[primary]['occurrences'] += \
                self.discovered_entities[secondary]['occurrences']
            
            # Update confidence
            self.entity_confidence[primary] = min(0.95, 
                (conf1 + conf2) / 2 + 0.1)
            
            # Remove secondary
            del self.discovered_entities[secondary]
            del self.entity_confidence[secondary]
    
    def get_knowledge_expansion_report(self) -> Dict:
        """Get report on knowledge expansion"""
        high_confidence_entities = [
            e for e, conf in self.entity_confidence.items() if conf > 0.7
        ]
        
        high_confidence_relationships = [
            rel for rel, conf in self.relationship_confidence.items() if conf > 0.7
        ]
        
        return {
            'learning_stats': self.learning_stats,
            'discovered_entities_count': len(self.discovered_entities),
            'discovered_relationships_count': len(self.discovered_relationships),
            'high_confidence_entities': len(high_confidence_entities),
            'high_confidence_relationships': len(high_confidence_relationships),
            'avg_entity_confidence': np.mean(list(self.entity_confidence.values())) 
                                     if self.entity_confidence else 0.0,
            'avg_relationship_confidence': np.mean(list(self.relationship_confidence.values()))
                                          if self.relationship_confidence else 0.0
        }
    
    def export_learned_knowledge(self, filepath: str = "models/learned_knowledge.json"):
        """Export learned knowledge for integration"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'discovered_entities': self.discovered_entities,
            'discovered_relationships': self.discovered_relationships,
            'entity_confidence': self.entity_confidence,
            'relationship_confidence': {
                f"{s}|{r}|{t}": conf 
                for (s, r, t), conf in self.relationship_confidence.items()
            },
            'learning_stats': self.learning_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[Self-Learning KG] Exported learned knowledge to {filepath}")


# =============================================================================
# 4. INTEGRATED RL AGENT WRAPPER
# =============================================================================

class ReinforcedLearningAgent:
    """
    Wraps existing agents with RL capabilities
    Learns optimal processing strategies
    """
    
    def __init__(self, base_agent, agent_type: str, rl_engine: ReinforcementLearningEngine):
        self.base_agent = base_agent
        self.agent_type = agent_type  # 'intent', 'entity', or 'answer'
        self.rl_engine = rl_engine
        
        # Current episode tracking
        self.current_episode = {
            'state': None,
            'action': None,
            'start_time': None
        }
        
        print(f"[RL Agent] Wrapped {agent_type} agent with RL")
    
    def process_with_rl(self, *args, **kwargs) -> Tuple[Any, str, str]:
        """
        Process with RL-guided action selection
        
        Returns:
            (result, state, action) tuple
        """
        # Create state representation
        if self.agent_type == 'intent':
            query = args[0] if args else kwargs.get('query', '')
            state = self.rl_engine.get_state_representation(
                query, 'unknown', [], {}
            )
        elif self.agent_type == 'entity':
            query = args[0] if args else kwargs.get('query', '')
            intent_info = args[1] if len(args) > 1 else kwargs.get('intent_info', {})
            state = self.rl_engine.get_state_representation(
                query, 
                intent_info.get('intent', 'unknown'),
                intent_info.get('entities_mentioned', []),
                {}
            )
        else:  # answer agent
            query = args[0] if args else kwargs.get('query', '')
            intent_info = args[1] if len(args) > 1 else kwargs.get('intent_info', {})
            entity_info = args[2] if len(args) > 2 else kwargs.get('entity_info', {})
            state = self.rl_engine.get_state_representation(
                query,
                intent_info.get('intent', 'unknown'),
                entity_info.get('entities', []),
                entity_info
            )
        
        # Select action using RL
        action = self.rl_engine.select_action(state, training=True)
        
        # Store episode info
        self.current_episode = {
            'state': state,
            'action': action,
            'start_time': time.time()
        }
        
        # Execute base agent with action guidance
        result = self._execute_with_action(action, *args, **kwargs)
        
        return result, state, action
    
    def _execute_with_action(self, action: str, *args, **kwargs):
        """Execute base agent with RL-guided action"""
        if action == "standard_processing":
            # Normal processing
            return self._call_base_agent(*args, **kwargs)
        
        elif action == "enhanced_extraction" and self.agent_type == 'entity':
            # More thorough entity extraction
            kwargs['enhanced_mode'] = True
            return self._call_base_agent(*args, **kwargs)
        
        elif action == "deep_database_query" and self.agent_type == 'entity':
            # Deeper database search
            kwargs['n_results'] = 20  # More results
            return self._call_base_agent(*args, **kwargs)
        
        elif action == "multi_pass_generation" and self.agent_type == 'answer':
            # Multiple generation passes
            result1 = self._call_base_agent(*args, **kwargs)
            # Second pass with refinement
            # ... (implementation depends on answer agent structure)
            return result1
        
        elif action == "confidence_boosting":
            # Boost confidence scores
            result = self._call_base_agent(*args, **kwargs)
            # Adjust confidence scores
            if isinstance(result, dict) and 'confidence' in result:
                result['confidence'] = min(1.0, result['confidence'] * 1.1)
            return result
        
        elif action == "context_expansion":
            # Expand context window
            kwargs['context_expansion'] = True
            return self._call_base_agent(*args, **kwargs)
        
        elif action == "relationship_inference":
            # Infer additional relationships
            result = self._call_base_agent(*args, **kwargs)
            # Add inferred relationships
            # ... (implementation depends on agent structure)
            return result
        
        else:
            # Fallback to standard
            return self._call_base_agent(*args, **kwargs)
    
    def _call_base_agent(self, *args, **kwargs):
        """Call the base agent's main method"""
        if self.agent_type == 'intent':
            return self.base_agent.analyze_intent(*args, **kwargs)
        elif self.agent_type == 'entity':
            return self.base_agent.extract_and_retrieve(*args, **kwargs)
        else:  # answer
            return self.base_agent.generate_answer(*args, **kwargs)
    
    def record_feedback(self, feedback_data: Dict):
        """Record feedback and update RL"""
        if not self.current_episode['state']:
            print("[RL Agent] No active episode to record feedback")
            return
        
        state = self.current_episode['state']
        action = self.current_episode['action']
        
        # Calculate reward
        reward = self.rl_engine.calculate_reward(feedback_data)
        
        # Create next state (simplified - in practice, capture actual next state)
        next_state = state  # Terminal state
        done = True
        
        # Update Q-value
        self.rl_engine.update_q_value(state, action, reward, next_state, done)
        
        # Store experience
        self.rl_engine.store_experience(state, action, reward, next_state, done)
        
        # Periodic experience replay
        if self.rl_engine.learning_stats['total_updates'] % 10 == 0:
            self.rl_engine.replay_experiences(batch_size=32)
        
        # Clear episode
        self.current_episode = {'state': None, 'action': None, 'start_time': None}
        
        print(f"[RL Agent] Recorded feedback for {self.agent_type} agent (reward={reward:.2f})")


# =============================================================================
# 5. CONTINUOUS LEARNING MANAGER
# =============================================================================

class ContinuousLearningManager:
    """
    Manages continuous learning across all systems
    Coordinates RL, self-learning vector DB, and knowledge graph
    """
    
    def __init__(self, rl_engine: ReinforcementLearningEngine,
                 vector_learner: SelfLearningVectorDB,
                 kg_learner: SelfLearningKnowledgeGraph):
        self.rl_engine = rl_engine
        self.vector_learner = vector_learner
        self.kg_learner = kg_learner
        
        # Learning schedule
        self.auto_save_interval = 300  # 5 minutes
        self.auto_retrain_interval = 3600  # 1 hour
        
        # Background learning thread
        self.learning_thread = None
        self.stop_learning = False
        
        # Performance tracking
        self.learning_history = []
        
        print("[Continuous Learning] Manager initialized")
    
    def start_background_learning(self):
        """Start background learning thread"""
        if self.learning_thread and self.learning_thread.is_alive():
            print("[Continuous Learning] Already running")
            return
        
        self.stop_learning = False
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        
        print("[Continuous Learning] Background learning started")
    
    def stop_background_learning(self):
        """Stop background learning thread"""
        self.stop_learning = True
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        
        print("[Continuous Learning] Background learning stopped")
    
    def _learning_loop(self):
        """Background learning loop"""
        last_save = time.time()
        last_retrain = time.time()
        
        while not self.stop_learning:
            try:
                current_time = time.time()
                
                # Auto-save models
                if current_time - last_save >= self.auto_save_interval:
                    self.save_all_models()
                    last_save = current_time
                
                # Auto-retrain
                if current_time - last_retrain >= self.auto_retrain_interval:
                    self.auto_retrain_all()
                    last_retrain = current_time
                
                # Sleep
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"[Continuous Learning] Error in learning loop: {e}")
                time.sleep(60)
    
    def record_interaction(self, query: str, intent: str, entities: List[str],
                          results: List[Dict], answer: str, 
                          feedback: Optional[Dict] = None):
        """Record complete interaction for learning"""
        # Vector DB learning
        if feedback:
            feedback_score = feedback.get('accuracy_score', 0.5) / 10
            was_helpful = feedback.get('approved', False)
            
            self.vector_learner.record_feedback(
                query, intent, results, feedback_score, was_helpful
            )
        
        # Knowledge graph learning
        self.kg_learner.extract_from_query_context(query, answer, results)
        
        # Infer new relationships periodically
        if len(self.kg_learner.discovered_entities) % 50 == 0:
            self.kg_learner.infer_missing_relationships()
        
        # Consolidate duplicates periodically
        if len(self.kg_learner.discovered_entities) % 100 == 0:
            self.kg_learner.consolidate_duplicate_entities()
        
        print("[Continuous Learning] Recorded interaction")
    
    def auto_retrain_all(self):
        """Auto-retrain all learning systems"""
        print("[Continuous Learning] Starting auto-retrain...")
        
        # Vector DB retraining
        vector_retrained = self.vector_learner.auto_retrain()
        
        # RL experience replay
        if len(self.rl_engine.experience_buffer) >= 32:
            self.rl_engine.replay_experiences(batch_size=64)
        
        # Knowledge graph inference
        inferred = self.kg_learner.infer_missing_relationships()
        
        # Track performance
        performance = {
            'timestamp': datetime.now().isoformat(),
            'vector_retrained': vector_retrained,
            'rl_experiences': len(self.rl_engine.experience_buffer),
            'kg_inferred': len(inferred),
            'rl_stats': self.rl_engine.get_learning_stats(),
            'vector_stats': self.vector_learner.get_performance_metrics(),
            'kg_stats': self.kg_learner.get_knowledge_expansion_report()
        }
        
        self.learning_history.append(performance)
        
        print(f"[Continuous Learning] Auto-retrain complete: "
              f"Vector={vector_retrained}, "
              f"RL experiences={len(self.rl_engine.experience_buffer)}, "
              f"KG inferred={len(inferred)}")
    
    def save_all_models(self):
        """Save all learning models"""
        print("[Continuous Learning] Saving all models...")
        
        self.rl_engine.save_model("models/rl_model.pkl")
        self.vector_learner.save_learning_data("models/vector_learning.json")
        self.kg_learner.export_learned_knowledge("models/learned_knowledge.json")
        
        # Save learning history
        with open("models/learning_history.json", 'w') as f:
            json.dump(self.learning_history[-1000:], f, indent=2)
        
        print("[Continuous Learning] All models saved")
    
    def load_all_models(self):
        """Load all learning models"""
        print("[Continuous Learning] Loading all models...")
        
        self.rl_engine.load_model("models/rl_model.pkl")
        self.vector_learner.load_learning_data("models/vector_learning.json")
        
        # Load learning history
        try:
            with open("models/learning_history.json", 'r') as f:
                self.learning_history = json.load(f)
        except FileNotFoundError:
            pass
        
        print("[Continuous Learning] All models loaded")
    
    def get_comprehensive_report(self) -> Dict:
        """Get comprehensive learning report"""
        return {
            'rl_stats': self.rl_engine.get_learning_stats(),
            'vector_stats': self.vector_learner.get_performance_metrics(),
            'kg_stats': self.kg_learner.get_knowledge_expansion_report(),
            'learning_history_count': len(self.learning_history),
            'background_learning_active': self.learning_thread and self.learning_thread.is_alive()
        }


# =============================================================================
# 6. INTEGRATION WITH EXISTING ORCHESTRATOR
# =============================================================================

# After your existing SimpleStateOrchestrator class, add this:

class RLEnhancedOrchestrator(SimpleStateOrchestrator):
    """
    RL-enhanced orchestrator that wraps existing agents
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize RL components
        self.rl_engine = ReinforcementLearningEngine()
        self.vector_learner = SelfLearningVectorDB(db_manager)
        self.kg_learner = SelfLearningKnowledgeGraph(knowledge_graph, db_manager)
        
        # Wrap agents with RL
        self.rl_intent_agent = ReinforcedLearningAgent(
            self.intent_agent, 'intent', self.rl_engine
        )
        self.rl_entity_agent = ReinforcedLearningAgent(
            self.entity_agent, 'entity', self.rl_engine
        )
        self.rl_answer_agent = ReinforcedLearningAgent(
            self.answer_agent, 'answer', self.rl_engine
        )
        
        # Continuous learning manager
        self.learning_manager = ContinuousLearningManager(
            self.rl_engine, self.vector_learner, self.kg_learner
        )
        
        # Load existing models
        self.learning_manager.load_all_models()
        
        # Start background learning
        self.learning_manager.start_background_learning()
        
        print("[RL Orchestrator] Enhanced orchestrator initialized with RL")
    
    @time_function
    def process_query_with_rl(self, query: str, chat_history: List = None,
                              documents_context: List = None,
                              user_profile: Dict = None) -> Dict[str, Any]:
        """Process query with RL-enhanced agents"""
        
        # Intent analysis with RL
        intent_result, intent_state, intent_action = self.rl_intent_agent.process_with_rl(query)
        
        # Entity extraction with RL + self-learning vector DB
        entity_result, entity_state, entity_action = self.rl_entity_agent.process_with_rl(
            query, intent_result, documents_context
        )
        
        # Use self-learning vector DB for enhanced retrieval
        if entity_result.get('entities'):
            enhanced_results = self.vector_learner.query_with_learning(
                query, intent_result.get('intent', 'general'), n_results=15
            )
            # Merge with existing results
            entity_result['enhanced_results'] = enhanced_results
        
        # Answer generation with RL
        answer_result, answer_state, answer_action = self.rl_answer_agent.process_with_rl(
            query, intent_result, entity_result, chat_history, documents_context
        )
        
        # Extract knowledge from this interaction
        self.learning_manager.record_interaction(
            query=query,
            intent=intent_result.get('intent', 'unknown'),
            entities=entity_result.get('entities', []),
            results=entity_result.get('enhanced_results', []),
            answer=answer_result
        )
        
        # Return results with RL metadata
        return {
            'answer': answer_result,
            'intent': intent_result,
            'entities': entity_result,
            'rl_metadata': {
                'intent_action': intent_action,
                'entity_action': entity_action,
                'answer_action': answer_action,
                'states': {
                    'intent': intent_state,
                    'entity': entity_state,
                    'answer': answer_state
                }
            }
        }
    
    def record_user_feedback(self, query: str, response_data: Dict, 
                           feedback: Dict):
        """Record user feedback for all learning systems"""
        
        # RL feedback for all agents
        self.rl_intent_agent.record_feedback(feedback)
        self.rl_entity_agent.record_feedback(feedback)
        self.rl_answer_agent.record_feedback(feedback)
        
        # Vector DB feedback
        feedback_score = feedback.get('accuracy_score', 0.5) / 10
        was_helpful = feedback.get('approved', False)
        
        self.vector_learner.record_feedback(
            query=query,
            intent=response_data.get('intent', {}).get('intent', 'unknown'),
            results=response_data.get('entities', {}).get('enhanced_results', []),
            feedback_score=feedback_score,
            was_helpful=was_helpful
        )
        
        print(f"[RL Orchestrator] Recorded user feedback (helpful={was_helpful})")
    
    def get_learning_dashboard(self) -> Dict:
        """Get comprehensive learning dashboard"""
        return self.learning_manager.get_comprehensive_report()


# =============================================================================
# 7. NEW API ENDPOINTS FOR LEARNING
# =============================================================================

# Add these routes to your Flask app:

@app.route("/api/rl/query", methods=["POST"])
def query_with_rl():
    """Query with RL-enhanced processing"""
    user = require_auth()
    if not user:
        return jsonify({"error": "User not authenticated"}), 401
    
    data = request.get_json()
    query = data.get("question", "").strip()
    
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400
    
    try:
        # Use RL-enhanced orchestrator
        result = rl_orchestrator.process_query_with_rl(
            query=query,
            chat_history=data.get("chat_history", []),
            documents_context=data.get("documents_context", []),
            user_profile={"user_id": user["sub"]}
        )
        
        return jsonify({
            "response": {"answer": result['answer']},
            "metadata": {
                "intent": result['intent'].get('intent', 'unknown'),
                "entities": result['entities'].get('entities', []),
                "rl_actions": result['rl_metadata']
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rl/feedback", methods=["POST"])
def submit_rl_feedback():
    """Submit feedback for reinforcement learning"""
    user = require_auth()
    if not user:
        return jsonify({"error": "User not authenticated"}), 401
    
    data = request.get_json()
    
    try:
        rl_orchestrator.record_user_feedback(
            query=data.get("query", ""),
            response_data=data.get("response_data", {}),
            feedback=data.get("feedback", {})
        )
        
        return jsonify({
            "success": True,
            "message": "Feedback recorded for learning"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/learning/dashboard", methods=["GET"])
def get_learning_dashboard():
    """Get learning system dashboard"""
    user = require_auth()
    if not user:
        return jsonify({"error": "User not authenticated"}), 401
    
    try:
        dashboard = rl_orchestrator.get_learning_dashboard()
        return jsonify(dashboard)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/learning/export", methods=["GET"])
def export_learned_knowledge():
    """Export all learned knowledge"""
    user = require_auth()
    if not user:
        return jsonify({"error": "User not authenticated"}), 401
    
    try:
        rl_orchestrator.learning_manager.save_all_models()
        
        return jsonify({
            "success": True,
            "message": "Learning models exported",
            "files": [
                "models/rl_model.pkl",
                "models/vector_learning.json",
                "models/learned_knowledge.json"
            ]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Initialize RL-enhanced orchestrator
rl_orchestrator = RLEnhancedOrchestrator()
print("✅ RL-Enhanced Orchestrator initialized with continuous learning")
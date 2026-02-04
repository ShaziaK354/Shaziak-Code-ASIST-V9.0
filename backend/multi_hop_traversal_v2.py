"""
Multi-Hop Graph Traversal Module for SAMM Agent
================================================
Version: 2.0.0
Date: 23-Jan-2026

NEW in v2.0.0 (Ticket 784 - Path Summaries):
- Path Summarization with concise LLM context generation
- SAMM Acronym Expansion integration
- SAMM Glossary Definition integration
- Token usage optimization
- Enhanced context formatting

Features (from v1.0.0):
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
✅ Path summarization working (NEW)
✅ Token usage optimized (NEW)
✅ Acronym expansion integrated (NEW)
✅ Glossary definitions available (NEW)
"""

import time
import logging
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MultiHopTraversal")


# =============================================================================
# SAMM ACRONYM EXPANDER (NEW)
# =============================================================================

class SAMMAcronymExpander:
    """
    Expands SAMM acronyms to their full forms.
    Helps make path summaries more understandable.
    """
    
    # Core SAMM acronyms (subset - full list can be loaded from JSON)
    DEFAULT_ACRONYMS = {
        # Organizations
        'DSCA': 'Defense Security Cooperation Agency',
        'DoD': 'Department of Defense',
        'DoS': 'Department of State',
        'MILDEP': 'Military Department',
        'IA': 'Implementing Agency',
        'SCO': 'Security Cooperation Organization',
        'DFAS': 'Defense Finance and Accounting Service',
        'DLA': 'Defense Logistics Agency',
        'DCMA': 'Defense Contract Management Agency',
        'DCAA': 'Defense Contract Audit Agency',
        'DASA DE&C': 'Deputy Assistant Secretary of the Army for Defense Exports and Cooperation',
        'SAF/IA': 'Deputy Under Secretary of the Air Force International Affairs',
        
        # Programs & Processes
        'FMS': 'Foreign Military Sales',
        'BPC': 'Building Partner Capacity',
        'IMET': 'International Military Education and Training',
        'EDA': 'Excess Defense Articles',
        'FMF': 'Foreign Military Financing',
        'MAP': 'Military Assistance Program',
        'EUM': 'End Use Monitoring',
        'CLSSA': 'Cooperative Logistics Supply Support Arrangement',
        
        # Documents & Systems
        'LOA': 'Letter of Offer and Acceptance',
        'LOR': 'Letter of Request',
        'SAMM': 'Security Assistance Management Manual',
        'DSAMS': 'Defense Security Assistance Management System',
        'MASL': 'Military Articles and Services List',
        'MAPAD': 'Military Assistance Program Address Directory',
        
        # Legal & Regulatory
        'AECA': 'Arms Export Control Act',
        'FAA': 'Foreign Assistance Act',
        'ITAR': 'International Traffic in Arms Regulations',
        'FAR': 'Federal Acquisition Regulation',
        
        # Personnel & Training
        'IMS': 'International Military Student',
        'IMSO': 'International Military Student Office',
        'MTT': 'Mobile Training Team',
        'ETSS': 'Extended Training Service Specialist',
        'FSP': 'Field Studies Program',
        
        # Financial
        'CAS': 'Contract Administrative Surcharge',
        'PC&H': 'Packing, Crating and Handling',
        'NRC': 'Nonrecurring Cost',
        'LSC': 'Logistics Support Charge',
        
        # Other Common
        'CONUS': 'Continental United States',
        'MDE': 'Major Defense Equipment',
        'SME': 'Significant Military Equipment',
        'CPI': 'Critical Program Information',
        'GFE': 'Government Furnished Equipment',
        'GFM': 'Government Furnished Material',
        
        # Leadership
        'USD(P)': 'Under Secretary of Defense for Policy',
        'SECDEF': 'Secretary of Defense',
        'ASD(ISP)': 'Assistant Secretary of Defense for International Security Policy',
        'DASD': 'Deputy Assistant Secretary of Defense',
    }
    
    def __init__(self, custom_acronyms: Dict[str, str] = None):
        """Initialize with default + custom acronyms."""
        self.acronyms = self.DEFAULT_ACRONYMS.copy()
        if custom_acronyms:
            self.acronyms.update(custom_acronyms)
        
        # Build case-insensitive lookup
        self._lookup = {k.lower(): v for k, v in self.acronyms.items()}
        
        logger.info(f"SAMMAcronymExpander initialized with {len(self.acronyms)} acronyms")
    
    def load_from_json(self, json_path: str) -> bool:
        """Load acronyms from JSON file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                self.acronyms.update(data)
            elif isinstance(data, list):
                # Format: [{"acronym": "ABC", "definition": "..."}, ...]
                for item in data:
                    if 'acronym' in item and 'definition' in item:
                        self.acronyms[item['acronym']] = item['definition']
            
            self._lookup = {k.lower(): v for k, v in self.acronyms.items()}
            logger.info(f"Loaded {len(data)} additional acronyms from {json_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Could not load acronyms from {json_path}: {e}")
            return False
    
    def expand(self, acronym: str) -> Optional[str]:
        """Get full form of an acronym."""
        return self._lookup.get(acronym.lower())
    
    def expand_text(self, text: str, first_occurrence_only: bool = True) -> str:
        """
        Expand acronyms in text.
        Format: "DSCA" -> "DSCA (Defense Security Cooperation Agency)"
        """
        expanded = set()
        result = text
        
        # Sort by length (longest first) to handle overlapping acronyms
        sorted_acronyms = sorted(self.acronyms.keys(), key=len, reverse=True)
        
        for acronym in sorted_acronyms:
            if first_occurrence_only and acronym.lower() in expanded:
                continue
            
            # Match whole words only
            pattern = r'\b' + re.escape(acronym) + r'\b'
            
            if re.search(pattern, result, re.IGNORECASE):
                full_form = self.acronyms[acronym]
                
                if first_occurrence_only:
                    # Replace first occurrence only
                    def replacer(match):
                        if acronym.lower() not in expanded:
                            expanded.add(acronym.lower())
                            return f"{match.group(0)} ({full_form})"
                        return match.group(0)
                    
                    result = re.sub(pattern, replacer, result, count=1, flags=re.IGNORECASE)
                else:
                    # Replace all occurrences
                    result = re.sub(
                        pattern,
                        lambda m: f"{m.group(0)} ({full_form})",
                        result,
                        flags=re.IGNORECASE
                    )
        
        return result
    
    def get_acronyms_in_text(self, text: str) -> List[Tuple[str, str]]:
        """Find all acronyms present in text with their definitions."""
        found = []
        text_lower = text.lower()
        
        for acronym, definition in self.acronyms.items():
            pattern = r'\b' + re.escape(acronym.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found.append((acronym, definition))
        
        return found


# =============================================================================
# SAMM GLOSSARY (NEW)
# =============================================================================

class SAMMGlossary:
    """
    Provides SAMM glossary definitions for key terms.
    Helps enrich path summaries with contextual definitions.
    """
    
    # Core glossary terms (subset - full list can be loaded from JSON)
    DEFAULT_GLOSSARY = {
        'letter of offer and acceptance': 'The legal instrument used by the USG to transfer itemized defense articles, defense services, and design and construction services to foreign partners.',
        'letter of request': 'A request from an eligible FMS participant country for the purchase of U.S. defense articles and services.',
        'foreign military sales': 'A process authorized by the Arms Export Control Act through which eligible foreign governments may purchase defense articles and services from the U.S. Government.',
        'building partner capacity': 'Security cooperation activities funded with USG appropriations to transfer defense articles and services to foreign partners.',
        'implementing agency': 'The military department or defense agency responsible for preparing a LOA and implementing an FMS case.',
        'defense article': 'Any weapon, weapons system, munition, or other item used for military assistance or sales.',
        'defense service': 'Any service, test, inspection, repair, training, or technical assistance used for military assistance or sales.',
        'end use monitoring': 'Activities to ensure defense articles transferred to foreign partners are used according to agreement terms.',
        'security cooperation organization': 'A DoD element located in a foreign country to carry out security cooperation responsibilities.',
        'case identifier': 'A unique six-digit identifier assigned to an FMS case for identification and accounting purposes.',
        'acceptance date': 'The date on which an authorized representative of the foreign partner signed the LOA document.',
        'implemented case': 'An FMS case signed by the foreign partner with initial deposit received and system transactions completed.',
        'closed case': 'An FMS case where all deliveries, services, and financial transactions are completed.',
    }
    
    def __init__(self, custom_glossary: Dict[str, str] = None):
        """Initialize with default + custom glossary."""
        self.glossary = self.DEFAULT_GLOSSARY.copy()
        if custom_glossary:
            self.glossary.update({k.lower(): v for k, v in custom_glossary.items()})
        
        logger.info(f"SAMMGlossary initialized with {len(self.glossary)} terms")
    
    def load_from_json(self, json_path: str) -> bool:
        """Load glossary from JSON file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                self.glossary.update({k.lower(): v for k, v in data.items()})
            elif isinstance(data, list):
                for item in data:
                    if 'term' in item and 'definition' in item:
                        self.glossary[item['term'].lower()] = item['definition']
            
            logger.info(f"Loaded {len(data)} glossary terms from {json_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Could not load glossary from {json_path}: {e}")
            return False
    
    def get_definition(self, term: str) -> Optional[str]:
        """Get definition for a term."""
        return self.glossary.get(term.lower())
    
    def find_relevant_terms(self, text: str, max_terms: int = 5) -> List[Tuple[str, str]]:
        """Find glossary terms relevant to the given text."""
        text_lower = text.lower()
        found = []
        
        for term, definition in self.glossary.items():
            if term in text_lower:
                found.append((term.title(), definition))
        
        # Sort by relevance (term frequency in text)
        found.sort(key=lambda x: text_lower.count(x[0].lower()), reverse=True)
        
        return found[:max_terms]


# =============================================================================
# PATH SUMMARIZER (NEW)
# =============================================================================

@dataclass
class PathSummaryConfig:
    """Configuration for path summarization."""
    max_summary_tokens: int = 500  # Approximate token limit for summary
    max_paths_to_summarize: int = 10  # Limit paths included in summary
    include_acronym_expansion: bool = True
    include_glossary_definitions: bool = True
    max_glossary_terms: int = 3
    summary_style: str = 'concise'  # 'concise', 'detailed', 'bullet'
    group_by_relationship: bool = True  # Group similar relationships
    

@dataclass
class PathSummary:
    """Represents a summarized path context for LLM."""
    summary_text: str
    expanded_acronyms: List[Tuple[str, str]]
    relevant_glossary: List[Tuple[str, str]]
    paths_count: int
    token_estimate: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'summary_text': self.summary_text,
            'expanded_acronyms': self.expanded_acronyms,
            'relevant_glossary': self.relevant_glossary,
            'paths_count': self.paths_count,
            'token_estimate': self.token_estimate,
            'metadata': self.metadata
        }


class PathSummarizer:
    """
    Generates concise summaries of graph paths for LLM context.
    
    Features:
    - Token-optimized output
    - Acronym expansion
    - Glossary integration
    - Multiple summary styles
    """
    
    def __init__(
        self,
        config: PathSummaryConfig = None,
        acronym_expander: SAMMAcronymExpander = None,
        glossary: SAMMGlossary = None
    ):
        self.config = config or PathSummaryConfig()
        self.acronym_expander = acronym_expander or SAMMAcronymExpander()
        self.glossary = glossary or SAMMGlossary()
        
        logger.info("PathSummarizer initialized")
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (words * 1.3)."""
        return int(len(text.split()) * 1.3)
    
    def summarize_paths(
        self,
        paths: List['MultiHopPath'],
        query: str,
        intent: str = None
    ) -> PathSummary:
        """
        Generate concise summary from paths.
        
        Args:
            paths: List of MultiHopPath objects
            query: Original user query
            intent: Query intent (authority, hierarchy, process, etc.)
            
        Returns:
            PathSummary with optimized context for LLM
        """
        if not paths:
            return PathSummary(
                summary_text="No relevant paths found.",
                expanded_acronyms=[],
                relevant_glossary=[],
                paths_count=0,
                token_estimate=5
            )
        
        # Limit paths to process
        paths_to_process = paths[:self.config.max_paths_to_summarize]
        
        # Generate summary based on style
        if self.config.summary_style == 'bullet':
            summary_text = self._generate_bullet_summary(paths_to_process, query, intent)
        elif self.config.summary_style == 'detailed':
            summary_text = self._generate_detailed_summary(paths_to_process, query, intent)
        else:
            summary_text = self._generate_concise_summary(paths_to_process, query, intent)
        
        # Find acronyms to expand
        expanded_acronyms = []
        if self.config.include_acronym_expansion:
            expanded_acronyms = self.acronym_expander.get_acronyms_in_text(summary_text)
        
        # Find relevant glossary terms
        relevant_glossary = []
        if self.config.include_glossary_definitions:
            relevant_glossary = self.glossary.find_relevant_terms(
                summary_text,
                max_terms=self.config.max_glossary_terms
            )
        
        # Calculate token estimate
        token_estimate = self.estimate_tokens(summary_text)
        
        # Add glossary definitions to context if under token limit
        glossary_text = ""
        if relevant_glossary and token_estimate < self.config.max_summary_tokens - 100:
            glossary_text = self._format_glossary_section(relevant_glossary)
            token_estimate += self.estimate_tokens(glossary_text)
        
        # Combine final text
        final_text = summary_text
        if glossary_text:
            final_text += "\n\n" + glossary_text
        
        return PathSummary(
            summary_text=final_text,
            expanded_acronyms=expanded_acronyms,
            relevant_glossary=relevant_glossary,
            paths_count=len(paths_to_process),
            token_estimate=token_estimate,
            metadata={
                'style': self.config.summary_style,
                'total_paths_available': len(paths),
                'query': query,
                'intent': intent
            }
        )
    
    def _generate_concise_summary(
        self,
        paths: List['MultiHopPath'],
        query: str,
        intent: str = None
    ) -> str:
        """Generate token-optimized concise summary."""
        parts = []
        
        # Header with context
        parts.append(f"[Context for: {query}]")
        parts.append("")
        
        # Group paths by relationship type for conciseness
        if self.config.group_by_relationship:
            relationship_groups = self._group_by_relationship(paths)
            
            for rel_type, group_paths in relationship_groups.items():
                if len(group_paths) == 1:
                    p = group_paths[0]
                    parts.append(f"• {p.start_node.upper()} → {p.end_node.upper()} ({rel_type})")
                else:
                    # Summarize group
                    starts = set(p.start_node.upper() for p in group_paths)
                    ends = set(p.end_node.upper() for p in group_paths)
                    parts.append(f"• {', '.join(starts)} → {', '.join(ends)} ({rel_type})")
        else:
            # Simple list
            for path in paths[:5]:
                summary_line = self._summarize_single_path(path, style='short')
                parts.append(f"• {summary_line}")
        
        # Add key findings
        findings = self._extract_key_findings(paths, intent)
        if findings:
            parts.append("")
            parts.append("Key findings:")
            for finding in findings[:3]:
                parts.append(f"  - {finding}")
        
        return "\n".join(parts)
    
    def _generate_bullet_summary(
        self,
        paths: List['MultiHopPath'],
        query: str,
        intent: str = None
    ) -> str:
        """Generate bullet-point summary."""
        parts = []
        
        parts.append(f"=== Path Summary: {query} ===")
        parts.append(f"Found {len(paths)} relevant paths")
        parts.append("")
        
        for i, path in enumerate(paths, 1):
            parts.append(f"{i}. {path.path_text}")
            if path.sections and any(path.sections):
                valid_sections = [s for s in path.sections if s]
                if valid_sections:
                    parts.append(f"   Reference: {', '.join(valid_sections[:2])}")
        
        return "\n".join(parts)
    
    def _generate_detailed_summary(
        self,
        paths: List['MultiHopPath'],
        query: str,
        intent: str = None
    ) -> str:
        """Generate detailed summary with reasoning chains."""
        parts = []
        
        parts.append(f"=== Detailed Path Analysis ===")
        parts.append(f"Query: {query}")
        parts.append(f"Paths analyzed: {len(paths)}")
        parts.append("")
        
        # Top 3 paths with full reasoning
        for i, path in enumerate(paths[:3], 1):
            parts.append(f"Path {i} (Score: {path.relevance_score:.3f}, {path.hops} hops):")
            parts.append(f"  Route: {path.path_text}")
            
            # Include reasoning steps
            if path.reasoning_chain:
                parts.append("  Reasoning:")
                for step in path.reasoning_chain[:3]:
                    parts.append(f"    → {step.explanation}")
            
            parts.append("")
        
        return "\n".join(parts)
    
    def _group_by_relationship(
        self,
        paths: List['MultiHopPath']
    ) -> Dict[str, List['MultiHopPath']]:
        """Group paths by their primary relationship type."""
        groups = {}
        
        for path in paths:
            # Use first relationship as primary
            primary_rel = path.relationships[0] if path.relationships else 'related_to'
            
            if primary_rel not in groups:
                groups[primary_rel] = []
            groups[primary_rel].append(path)
        
        return groups
    
    def _summarize_single_path(
        self,
        path: 'MultiHopPath',
        style: str = 'short'
    ) -> str:
        """Summarize a single path."""
        if style == 'short':
            # Ultra-concise: "A → B via relationship"
            rel = path.relationships[0] if path.relationships else "→"
            return f"{path.start_node.upper()} → {path.end_node.upper()} ({rel})"
        else:
            # Full path
            return path.path_text
    
    def _extract_key_findings(
        self,
        paths: List['MultiHopPath'],
        intent: str = None
    ) -> List[str]:
        """Extract key findings from paths."""
        findings = []
        
        # Count relationship types
        rel_counts = {}
        for path in paths:
            for rel in path.relationships:
                rel_counts[rel] = rel_counts.get(rel, 0) + 1
        
        # Top relationships
        top_rels = sorted(rel_counts.items(), key=lambda x: x[1], reverse=True)[:2]
        
        for rel, count in top_rels:
            if count > 1:
                findings.append(f"Multiple {rel} relationships found ({count} instances)")
        
        # Authority-specific findings
        if intent and 'authority' in intent.lower():
            authority_rels = ['supervised_by', 'supervises', 'reports_to', 'authorizes']
            auth_paths = [p for p in paths if any(r in authority_rels for r in p.relationships)]
            if auth_paths:
                top_auth = auth_paths[0]
                findings.append(f"Authority chain: {top_auth.start_node} → {top_auth.end_node}")
        
        return findings
    
    def _format_glossary_section(
        self,
        terms: List[Tuple[str, str]]
    ) -> str:
        """Format glossary terms section."""
        if not terms:
            return ""
        
        parts = ["[Key Terms]"]
        for term, definition in terms:
            # Truncate long definitions
            short_def = definition[:150] + "..." if len(definition) > 150 else definition
            parts.append(f"• {term}: {short_def}")
        
        return "\n".join(parts)
    
    def get_llm_context(
        self,
        paths: List['MultiHopPath'],
        query: str,
        intent: str = None,
        max_tokens: int = None
    ) -> str:
        """
        Get optimized context string for LLM prompt.
        
        This is the main method to call for RAG integration.
        """
        if max_tokens:
            self.config.max_summary_tokens = max_tokens
        
        summary = self.summarize_paths(paths, query, intent)
        
        # Build final context
        context_parts = []
        
        # Main summary
        context_parts.append(summary.summary_text)
        
        # Acronym legend (if any found)
        if summary.expanded_acronyms:
            context_parts.append("")
            context_parts.append("[Acronyms]")
            for acronym, full in summary.expanded_acronyms[:5]:
                context_parts.append(f"• {acronym} = {full}")
        
        return "\n".join(context_parts)


# =============================================================================
# CONFIGURATION (from v1.0.0)
# =============================================================================

@dataclass
class MultiHopConfig:
    """Configuration for multi-hop graph traversal."""
    
    # Hop limits
    min_hops: int = 1
    max_hops: int = 5
    default_hops: int = 3
    
    # Path limits
    max_paths_per_entity: int = 20
    max_total_paths: int = 50
    
    # Performance limits
    max_traversal_time_ms: int = 5000
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    
    # Scoring weights
    hop_decay_factor: float = 0.8
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
    
    # NEW: Summarization settings
    enable_path_summarization: bool = True
    summary_config: PathSummaryConfig = field(default_factory=PathSummaryConfig)
    
    def validate(self):
        """Validate configuration values."""
        if self.max_hops < self.min_hops:
            raise ValueError(f"max_hops ({self.max_hops}) must be >= min_hops ({self.min_hops})")
        if self.max_hops > 10:
            logger.warning(f"max_hops={self.max_hops} is very high, may cause performance issues")
        if self.hop_decay_factor <= 0 or self.hop_decay_factor > 1:
            raise ValueError("hop_decay_factor must be between 0 and 1")


DEFAULT_CONFIG = MultiHopConfig()


# =============================================================================
# DATA CLASSES (from v1.0.0)
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
    # NEW: Path summary
    path_summary: Optional[PathSummary] = None
    
    def to_dict(self) -> Dict:
        result = {
            'query': self.query,
            'entities': self.entities,
            'paths': [p.to_dict() for p in self.paths],
            'metrics': self.metrics.to_dict(),
            'context_text': self.context_text,
            'reasoning_summary': self.reasoning_summary,
            'config': self.config_used
        }
        if self.path_summary:
            result['path_summary'] = self.path_summary.to_dict()
        return result


# =============================================================================
# PATH RELEVANCE SCORER (from v1.0.0)
# =============================================================================

class PathRelevanceScorer:
    """Scores multi-hop paths based on relevance to query."""
    
    def __init__(self, config: MultiHopConfig = None):
        self.config = config or DEFAULT_CONFIG
        
        self.intent_keywords = {
            'authority': ['supervise', 'supervision', 'authority', 'oversee', 'manage', 'direct', 'control'],
            'hierarchy': ['report', 'chain', 'structure', 'organization', 'part of'],
            'process': ['process', 'procedure', 'step', 'workflow', 'how'],
            'definition': ['what is', 'define', 'meaning', 'explain'],
            'relationship': ['related', 'connect', 'between', 'link', 'coordinate']
        }
    
    def score_path(self, path: MultiHopPath, query: str, intent: str = None) -> float:
        """Calculate relevance score for a path."""
        score = 1.0
        
        # 1. Hop decay
        hop_penalty = self.config.hop_decay_factor ** path.hops
        score *= hop_penalty
        
        # 2. Relationship type weights
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
        
        query_words = set(query_lower.split())
        for node in path.nodes:
            node_words = set(node.lower().replace('_', ' ').split())
            overlap = query_words.intersection(node_words)
            if overlap:
                query_boost += 0.1 * len(overlap)
        
        if intent:
            intent_lower = intent.lower()
            for intent_type, keywords in self.intent_keywords.items():
                if intent_type in intent_lower:
                    for rel in path.relationships:
                        if any(kw in rel.lower() for kw in keywords):
                            query_boost += 0.15
        
        score *= min(query_boost, 2.0)
        
        # 4. Section reference bonus
        valid_sections = [s for s in path.sections if s and s.strip()]
        if valid_sections:
            section_bonus = 1.0 + (0.05 * len(valid_sections))
            score *= min(section_bonus, 1.25)
        
        return min(score, 1.0)
    
    def rank_paths(self, paths: List[MultiHopPath], query: str, intent: str = None) -> List[MultiHopPath]:
        """Rank paths by relevance score."""
        for path in paths:
            path.relevance_score = self.score_path(path, query, intent)
        
        return sorted(paths, key=lambda p: p.relevance_score, reverse=True)


# =============================================================================
# MULTI-HOP GRAPH TRAVERSER (Enhanced in v2.0.0)
# =============================================================================

class MultiHopGraphTraverser:
    """
    Enhanced multi-hop graph traversal with:
    - Configurable hop limits (3+ hops supported)
    - Reasoning chain generation
    - Intermediate node logging
    - Path relevance scoring
    - Path summarization (NEW in v2.0.0)
    - Acronym expansion (NEW in v2.0.0)
    - Glossary integration (NEW in v2.0.0)
    """
    
    def __init__(self, config: MultiHopConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.config.validate()
        
        self.scorer = PathRelevanceScorer(self.config)
        
        # NEW: Path summarization components
        self.summarizer = PathSummarizer(
            config=self.config.summary_config,
            acronym_expander=SAMMAcronymExpander(),
            glossary=SAMMGlossary()
        )
        
        # Graph data structures
        self.relationship_graph: Dict[str, List[Dict]] = {}
        self.entity_info: Dict[str, Dict] = {}
        self.entities: List[Dict] = []
        self.relationships: List[Dict] = []
        
        # Cache for performance
        self._path_cache: Dict[str, Tuple[List[MultiHopPath], float]] = {}
        
        # Logging
        self._traversal_log: List[Dict] = []
        
        logger.info(f"MultiHopGraphTraverser v2.0 initialized with max_hops={self.config.max_hops}")
    
    def load_from_json(self, json_path: str) -> bool:
        """Load knowledge graph from JSON file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.relationships = data.get('relationships', [])
            entities_data = data.get('entities', [])
            
            if isinstance(entities_data, list):
                self.entities = entities_data
            elif isinstance(entities_data, dict):
                self.entities = []
                for category, category_entities in entities_data.items():
                    if isinstance(category_entities, dict):
                        for entity_id, entity_data in category_entities.items():
                            entity = {
                                'id': entity_data.get('id', entity_id),
                                'name': entity_data.get('label', entity_data.get('name', entity_id)),
                                'type': entity_data.get('type', category.rstrip('s')),
                                'properties': {
                                    'definition': entity_data.get('definition', ''),
                                    'section': entity_data.get('section', ''),
                                    'category': category
                                }
                            }
                            for key, value in entity_data.items():
                                if key not in ['id', 'label', 'name', 'type', 'definition', 'section']:
                                    entity['properties'][key] = value
                            self.entities.append(entity)
                    elif isinstance(category_entities, list):
                        for entity_data in category_entities:
                            if isinstance(entity_data, dict):
                                entity = {
                                    'id': entity_data.get('id', ''),
                                    'name': entity_data.get('label', entity_data.get('name', '')),
                                    'type': entity_data.get('type', category.rstrip('s')),
                                    'properties': entity_data
                                }
                                self.entities.append(entity)
            
            self._build_graph()
            logger.info(f"Loaded {len(self.entities)} entities, {len(self.relationships)} relationships")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load JSON: {e}")
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
        
        for entity in self.entities:
            entity_id = entity.get('id', '').lower()
            if entity_id:
                self.entity_info[entity_id] = {
                    'name': entity.get('name', entity_id),
                    'type': entity.get('type', 'unknown'),
                    'properties': entity.get('properties', {})
                }
        
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
        """Find all multi-hop paths from entity using DFS."""
        start_time = time.time()
        
        max_hops = max_hops if max_hops is not None else self.config.default_hops
        max_paths = max_paths or self.config.max_paths_per_entity
        max_hops = min(max_hops, self.config.max_hops)
        
        metrics = TraversalMetrics()
        paths: List[MultiHopPath] = []
        
        if max_hops <= 0:
            return paths, metrics
        
        entity_lower = entity.lower()
        
        # Check cache
        cache_key = f"{entity_lower}:{max_hops}"
        if self.config.enable_caching and cache_key in self._path_cache:
            cached_paths, cache_time = self._path_cache[cache_key]
            if time.time() - cache_time < self.config.cache_ttl_seconds:
                metrics.cache_hits += 1
                return cached_paths, metrics
        
        metrics.cache_misses += 1
        
        if entity_lower not in self.relationship_graph:
            logger.warning(f"Entity '{entity}' not found in graph")
            return paths, metrics
        
        if self.config.log_intermediate_nodes:
            self._log_traversal_event("START", {'entity': entity, 'max_hops': max_hops})
        
        stack = [(entity_lower, [entity_lower], [], [], 0, [])]
        visited_paths: Set[str] = set()
        path_counter = 0
        
        all_paths: List[MultiHopPath] = []
        paths_per_depth: Dict[int, int] = {i: 0 for i in range(max_hops + 1)}
        max_per_depth = 100
        max_total_paths = 600
        
        while stack and len(all_paths) < max_total_paths:
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.config.max_traversal_time_ms:
                metrics.timeout_occurred = True
                break
            
            current, path_nodes, path_rels, path_sections, depth, intermediates = stack.pop()
            metrics.nodes_visited += 1
            metrics.max_depth_reached = max(metrics.max_depth_reached, depth)
            
            if depth >= max_hops:
                continue
            
            if current in self.relationship_graph:
                edges = self.relationship_graph[current][:50]
                
                for edge in edges:
                    target = edge['target']
                    rel_type = edge['type']
                    section = edge.get('section', '')
                    
                    if target in path_nodes:
                        continue
                    
                    new_path_nodes = path_nodes + [target]
                    new_path_rels = path_rels + [rel_type]
                    new_path_sections = path_sections + [section]
                    new_depth = depth + 1
                    
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
                    
                    if self.config.log_intermediate_nodes:
                        self._log_traversal_event("HOP", {
                            'hop': new_depth, 'from': current, 'to': target,
                            'relationship': rel_type, 'section': section
                        })
                    
                    path_text = f"{entity.upper()}"
                    for i, rel in enumerate(new_path_rels):
                        path_text += f" --[{rel}]--> {new_path_nodes[i+1].upper()}"
                    
                    path_key = " -> ".join(new_path_nodes)
                    
                    if path_key not in visited_paths:
                        visited_paths.add(path_key)
                        path_counter += 1
                        paths_per_depth[new_depth] = paths_per_depth.get(new_depth, 0) + 1
                        
                        reasoning_chain = self._build_reasoning_chain(
                            entity_lower, new_path_nodes, new_path_rels, new_path_sections
                        )
                        
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
                    
                    if new_depth < max_hops and paths_per_depth.get(new_depth, 0) < max_per_depth:
                        stack.append((target, new_path_nodes, new_path_rels, new_path_sections, new_depth, new_intermediates))
        
        # Apply relevance scoring
        if query:
            all_paths = self.scorer.rank_paths(all_paths, query, intent)
        else:
            all_paths.sort(key=lambda x: x.hops)
        
        # Diversity: ensure paths from different hop levels
        paths_by_hops: Dict[int, List[MultiHopPath]] = {}
        for p in all_paths:
            if p.hops not in paths_by_hops:
                paths_by_hops[p.hops] = []
            paths_by_hops[p.hops].append(p)
        
        paths = []
        if paths_by_hops:
            hop_levels = sorted(paths_by_hops.keys())
            paths_per_level = max(3, max_paths // len(hop_levels))
            
            for hop in hop_levels:
                level_paths = paths_by_hops[hop][:paths_per_level]
                paths.extend(level_paths)
            
            remaining = max_paths - len(paths)
            if remaining > 0:
                used_ids = {p.path_id for p in paths}
                for p in all_paths:
                    if p.path_id not in used_ids:
                        paths.append(p)
                        if len(paths) >= max_paths:
                            break
        
        paths.sort(key=lambda x: x.relevance_score, reverse=True)
        paths = paths[:max_paths]
        metrics.paths_after_scoring = len(paths)
        metrics.total_time_ms = (time.time() - start_time) * 1000
        
        if self.config.enable_caching:
            self._path_cache[cache_key] = (paths, time.time())
        
        if self.config.log_intermediate_nodes:
            self._log_traversal_event("COMPLETE", {'paths_found': len(paths), 'time_ms': metrics.total_time_ms})
        
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
            
            from_info = self.entity_info.get(from_node, {})
            to_info = self.entity_info.get(to_node, {})
            
            explanation = self._generate_step_explanation(
                from_node, to_node, rel,
                from_info.get('type', 'entity'),
                to_info.get('type', 'entity')
            )
            
            confidence = self.config.relationship_type_weights.get(
                rel.lower().replace(' ', '_'), 0.5
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
        self, from_node: str, to_node: str, relationship: str,
        from_type: str, to_type: str
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
            event = {'timestamp': time.time(), 'type': event_type, 'data': data}
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
        Now includes path summarization for optimized LLM context.
        """
        start_time = time.time()
        max_hops = max_hops or self.config.default_hops
        
        all_paths: List[MultiHopPath] = []
        combined_metrics = TraversalMetrics()
        
        for entity in entities:
            paths, metrics = self.find_multi_hop_paths(
                entity=entity, max_hops=max_hops, query=query, intent=intent
            )
            all_paths.extend(paths)
            
            combined_metrics.nodes_visited += metrics.nodes_visited
            combined_metrics.paths_found += metrics.paths_found
            combined_metrics.cache_hits += metrics.cache_hits
            combined_metrics.cache_misses += metrics.cache_misses
            combined_metrics.max_depth_reached = max(combined_metrics.max_depth_reached, metrics.max_depth_reached)
            if metrics.timeout_occurred:
                combined_metrics.timeout_occurred = True
        
        # Deduplicate and re-rank
        seen_paths: Set[str] = set()
        unique_paths: List[MultiHopPath] = []
        for path in all_paths:
            if path.path_text not in seen_paths:
                seen_paths.add(path.path_text)
                unique_paths.append(path)
        
        unique_paths = self.scorer.rank_paths(unique_paths, query, intent)
        unique_paths = unique_paths[:self.config.max_total_paths]
        
        combined_metrics.paths_after_scoring = len(unique_paths)
        combined_metrics.total_time_ms = (time.time() - start_time) * 1000
        
        # Build context text (legacy format)
        context_text = self._build_context_text(unique_paths, query, intent)
        
        # Build reasoning summary
        reasoning_summary = self._build_reasoning_summary(unique_paths)
        
        # NEW: Generate optimized path summary
        path_summary = None
        if self.config.enable_path_summarization:
            path_summary = self.summarizer.summarize_paths(unique_paths, query, intent)
        
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
                'scoring_enabled': True,
                'summarization_enabled': self.config.enable_path_summarization
            },
            path_summary=path_summary
        )
    
    def get_optimized_llm_context(
        self,
        entities: List[str],
        query: str,
        intent: str = None,
        max_tokens: int = 500
    ) -> str:
        """
        NEW: Get token-optimized context for LLM prompt.
        
        Use this method for RAG integration where token efficiency matters.
        """
        result = self.get_context_for_query(entities, query, intent)
        
        return self.summarizer.get_llm_context(
            result.paths, query, intent, max_tokens=max_tokens
        )
    
    def _build_context_text(
        self,
        paths: List[MultiHopPath],
        query: str,
        intent: str = None
    ) -> str:
        """Build context text for LLM prompt (legacy format)."""
        parts = []
        
        parts.append(f"=== MULTI-HOP GRAPH CONTEXT (up to {self.config.default_hops} hops) ===")
        parts.append(f"Query: {query}")
        parts.append(f"Paths found: {len(paths)}")
        parts.append("")
        
        hop_groups: Dict[int, List[MultiHopPath]] = {}
        for path in paths:
            if path.hops not in hop_groups:
                hop_groups[path.hops] = []
            hop_groups[path.hops].append(path)
        
        for hop_count in sorted(hop_groups.keys()):
            parts.append(f"--- {hop_count}-HOP PATHS ---")
            for path in hop_groups[hop_count][:5]:
                parts.append(f"• {path.path_text}")
                parts.append(f"  Score: {path.relevance_score:.3f}")
                
                valid_sections = [s for s in path.sections if s and s.strip()]
                if valid_sections:
                    parts.append(f"  Sections: {', '.join(valid_sections)}")
                
                if path.reasoning_chain and self.config.log_reasoning_chain:
                    parts.append(f"  Reasoning: {' → '.join(s.relationship for s in path.reasoning_chain)}")
                
                parts.append("")
        
        return '\n'.join(parts)
    
    def _build_reasoning_summary(self, paths: List[MultiHopPath]) -> str:
        """Build overall reasoning summary."""
        if not paths:
            return "No paths found."
        
        parts = ["=== REASONING CHAIN SUMMARY ==="]
        
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
    enable_summarization: bool = True,
    **config_kwargs
) -> MultiHopGraphTraverser:
    """
    Factory function to create configured MultiHopGraphTraverser.
    
    Usage:
        # From JSON file with summarization
        traverser = create_multi_hop_traverser(
            json_path="knowledge_graph.json",
            enable_summarization=True
        )
        
        # Get optimized context for RAG
        context = traverser.get_optimized_llm_context(
            entities=["dsca"],
            query="Who supervises DSCA?",
            max_tokens=500
        )
    """
    summary_config = PathSummaryConfig() if enable_summarization else PathSummaryConfig()
    
    config = MultiHopConfig(
        max_hops=max_hops,
        enable_path_summarization=enable_summarization,
        summary_config=summary_config,
        **config_kwargs
    )
    
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
    """Backward-compatible wrapper maintaining existing API."""
    
    def __init__(self, traverser: MultiHopGraphTraverser):
        self.traverser = traverser
        self.entities = traverser.entities
        self.relationships = traverser.relationships
        self.relationship_graph = traverser.relationship_graph
    
    def find_2hop_paths(self, entity: str, max_paths: int = 10) -> List[Dict]:
        """Backward compatible 2-hop function."""
        paths, _ = self.traverser.find_multi_hop_paths(
            entity=entity, max_hops=2, max_paths=max_paths
        )
        return [
            {
                'hops': p.hops, 'path': p.nodes, 'relationships': p.relationships,
                'path_text': p.path_text, 'sections': p.sections
            }
            for p in paths
        ]
    
    def find_nhop_paths(self, entity: str, max_hops: int = 3, max_paths: int = 15) -> List[Dict]:
        """Backward compatible n-hop function."""
        paths, _ = self.traverser.find_multi_hop_paths(
            entity=entity, max_hops=max_hops, max_paths=max_paths
        )
        return [
            {
                'hops': p.hops, 'path': p.nodes, 'relationships': p.relationships,
                'path_text': p.path_text, 'sections': p.sections
            }
            for p in paths
        ]
    
    def get_context_for_query(
        self, entities: List[str], query: str, intent: str = None
    ) -> Dict:
        """Backward compatible context function."""
        result = self.traverser.get_context_for_query(entities, query, intent)
        
        return {
            'paths': [
                {
                    'hops': p.hops, 'path': p.nodes, 'relationships': p.relationships,
                    'path_text': p.path_text, 'sections': p.sections
                }
                for p in result.paths
            ],
            'authority_chains': {},
            'context_text': result.context_text,
            'is_authority_question': 'authority' in query.lower() or 'supervise' in query.lower(),
            'relationship_count': len(result.paths),
            'reasoning_summary': result.reasoning_summary,
            'metrics': result.metrics.to_dict(),
            # NEW: Optimized summary
            'optimized_summary': result.path_summary.summary_text if result.path_summary else None
        }


# =============================================================================
# MAIN - FOR TESTING
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Multi-Hop Graph Traversal Module v2.0 - Test Run")
    print("="*60)
    
    # Test acronym expander
    print("\n📚 Testing Acronym Expander...")
    expander = SAMMAcronymExpander()
    test_text = "DSCA manages FMS cases through the LOA process."
    expanded = expander.expand_text(test_text)
    print(f"Original: {test_text}")
    print(f"Expanded: {expanded}")
    
    # Test glossary
    print("\n📖 Testing Glossary...")
    glossary = SAMMGlossary()
    terms = glossary.find_relevant_terms("letter of offer and acceptance for foreign military sales")
    print(f"Found terms: {[t[0] for t in terms]}")
    
    # Test path summarizer
    print("\n📝 Testing Path Summarizer...")
    summarizer = PathSummarizer()
    
    # Create mock path
    mock_path = MultiHopPath(
        path_id="test_1",
        start_node="DSCA",
        end_node="USD(P)",
        hops=2,
        nodes=["dsca", "osd", "usd(p)"],
        relationships=["reports_to", "part_of"],
        sections=["C1.3.1", "C1.2"],
        intermediate_nodes=[],
        reasoning_chain=[],
        relevance_score=0.85,
        path_text="DSCA --[reports_to]--> OSD --[part_of]--> USD(P)"
    )
    
    summary = summarizer.summarize_paths([mock_path], "Who supervises DSCA?", "authority")
    print(f"Summary:\n{summary.summary_text}")
    print(f"Token estimate: {summary.token_estimate}")
    
    print("\n" + "="*60)
    print("Module v2.0 ready for integration!")
    print("="*60)

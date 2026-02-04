"""
Step 1: Extract Training Data from Graph RAG System
===================================================
Extracts knowledge from Cosmos Gremlin Graph DB and ChromaDB Vector Store
to create training data for QLoRA fine-tuning.

Usage:
    python extract_graph_data.py --output ./samm_raw_data
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import hashlib

# Database clients
try:
    from gremlin_python.driver import client, serializer
    GREMLIN_AVAILABLE = True
except ImportError:
    GREMLIN_AVAILABLE = False
    print("⚠️ Gremlin client not available - install with: pip install gremlinpython")

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("⚠️ ChromaDB not available - install with: pip install chromadb")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    # Cosmos Gremlin
    gremlin_endpoint: str = os.getenv("COSMOS_GREMLIN_ENDPOINT", "asist-graph-db.gremlin.cosmos.azure.com")
    gremlin_database: str = os.getenv("COSMOS_GREMLIN_DATABASE", "ASIST-Agent-1.1DB")
    gremlin_graph: str = os.getenv("COSMOS_GREMLIN_COLLECTION", "AGENT1.4")
    gremlin_key: str = os.getenv("COSMOS_GREMLIN_KEY", "")
    
    # ChromaDB
    chroma_path: str = os.getenv("VECTOR_DB_PATH", "./vector_db")
    chroma_collection: str = os.getenv("VECTOR_DB_COLLECTION", "samm_all_chapters")
    
    # HITL Corrections
    hitl_file: str = "./hitl_corrections.json"

@dataclass
class TrainingExample:
    """A single training example"""
    id: str
    instruction: str
    input: str
    output: str
    source: str  # 'graph', 'vector', 'hitl'
    metadata: Dict[str, Any]

# ============================================================================
# DATA EXTRACTORS
# ============================================================================

class GremlinDataExtractor:
    """Extract knowledge from Cosmos Gremlin Graph Database"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.client = None
        
    def connect(self) -> bool:
        """Connect to Cosmos Gremlin"""
        if not GREMLIN_AVAILABLE:
            print("❌ Gremlin client not installed")
            return False
            
        if not self.config.gremlin_key:
            print("❌ Gremlin key not configured")
            return False
            
        try:
            username = f"/dbs/{self.config.gremlin_database}/colls/{self.config.gremlin_graph}"
            endpoint_url = f"wss://{self.config.gremlin_endpoint}:443/gremlin"
            
            self.client = client.Client(
                url=endpoint_url,
                traversal_source="g",
                username=username,
                password=self.config.gremlin_key,
                message_serializer=serializer.GraphSONSerializersV2d0()
            )
            
            # Test connection
            result = self.client.submit("g.V().count()").all().result()
            print(f"✅ Connected to Gremlin - {result[0]} vertices")
            return True
            
        except Exception as e:
            print(f"❌ Gremlin connection failed: {e}")
            return False
    
    def extract_entities(self) -> List[Dict]:
        """Extract all entities (vertices) from graph"""
        if not self.client:
            return []
            
        entities = []
        
        try:
            # Get all vertices with their properties
            query = """
            g.V().project('id', 'label', 'properties')
              .by(id())
              .by(label())
              .by(valueMap())
            """
            
            results = self.client.submit(query).all().result()
            
            for vertex in results:
                entity = {
                    'id': vertex.get('id', ''),
                    'label': vertex.get('label', ''),
                    'properties': {}
                }
                
                # Flatten properties
                props = vertex.get('properties', {})
                for key, values in props.items():
                    if isinstance(values, list) and len(values) > 0:
                        entity['properties'][key] = values[0]
                    else:
                        entity['properties'][key] = values
                        
                entities.append(entity)
                
            print(f"  📊 Extracted {len(entities)} entities from graph")
            
        except Exception as e:
            print(f"  ❌ Error extracting entities: {e}")
            
        return entities
    
    def extract_relationships(self) -> List[Dict]:
        """Extract all relationships (edges) from graph"""
        if not self.client:
            return []
            
        relationships = []
        
        try:
            query = """
            g.E().project('id', 'label', 'from', 'to', 'properties')
              .by(id())
              .by(label())
              .by(outV().values('name').fold())
              .by(inV().values('name').fold())
              .by(valueMap())
            """
            
            results = self.client.submit(query).all().result()
            
            for edge in results:
                relationship = {
                    'id': edge.get('id', ''),
                    'type': edge.get('label', ''),
                    'from': edge.get('from', ['Unknown'])[0] if edge.get('from') else 'Unknown',
                    'to': edge.get('to', ['Unknown'])[0] if edge.get('to') else 'Unknown',
                    'properties': edge.get('properties', {})
                }
                relationships.append(relationship)
                
            print(f"  🔗 Extracted {len(relationships)} relationships from graph")
            
        except Exception as e:
            print(f"  ❌ Error extracting relationships: {e}")
            
        return relationships
    
    def generate_training_examples(self, entities: List[Dict], relationships: List[Dict]) -> List[TrainingExample]:
        """Generate training examples from graph data"""
        examples = []
        
        # Entity definition examples
        for entity in entities:
            name = entity['properties'].get('name', entity['id'])
            definition = entity['properties'].get('definition', '')
            section = entity['properties'].get('section', '')
            role = entity['properties'].get('role', '')
            
            if definition or role:
                example_id = hashlib.md5(f"entity_{entity['id']}".encode()).hexdigest()[:8]
                
                examples.append(TrainingExample(
                    id=f"graph_entity_{example_id}",
                    instruction=f"What is {name}?",
                    input="",
                    output=self._format_entity_answer(name, definition, role, section),
                    source="graph",
                    metadata={
                        "entity_id": entity['id'],
                        "entity_type": entity['label'],
                        "section": section
                    }
                ))
        
        # Relationship examples
        for rel in relationships:
            example_id = hashlib.md5(f"rel_{rel['id']}".encode()).hexdigest()[:8]
            
            # Generate Q&A about relationship
            if rel['type'] == 'supervises':
                examples.append(TrainingExample(
                    id=f"graph_rel_{example_id}",
                    instruction=f"Who supervises {rel['to']}?",
                    input="",
                    output=f"{rel['from']} supervises {rel['to']}.",
                    source="graph",
                    metadata={"relationship_type": rel['type']}
                ))
            elif rel['type'] == 'isSubsetOf':
                examples.append(TrainingExample(
                    id=f"graph_rel_{example_id}",
                    instruction=f"What is the relationship between {rel['from']} and {rel['to']}?",
                    input="",
                    output=f"{rel['from']} is a subset of {rel['to']}.",
                    source="graph",
                    metadata={"relationship_type": rel['type']}
                ))
        
        print(f"  📝 Generated {len(examples)} training examples from graph")
        return examples
    
    def _format_entity_answer(self, name: str, definition: str, role: str, section: str) -> str:
        """Format an entity answer"""
        answer = f"{name}"
        
        if definition:
            answer += f" is defined as: {definition}"
        elif role:
            answer += f" {role}"
        
        if section:
            answer += f" (SAMM Section {section})"
        
        return answer
    
    def close(self):
        """Close connection"""
        if self.client:
            self.client.close()


class ChromaDataExtractor:
    """Extract knowledge from ChromaDB Vector Store"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.client = None
        
    def connect(self) -> bool:
        """Connect to ChromaDB"""
        if not CHROMA_AVAILABLE:
            print("❌ ChromaDB not installed")
            return False
            
        try:
            if Path(self.config.chroma_path).exists():
                self.client = chromadb.PersistentClient(path=self.config.chroma_path)
                collections = self.client.list_collections()
                print(f"✅ Connected to ChromaDB - {len(collections)} collections")
                return True
            else:
                print(f"❌ ChromaDB path not found: {self.config.chroma_path}")
                return False
                
        except Exception as e:
            print(f"❌ ChromaDB connection failed: {e}")
            return False
    
    def extract_documents(self, collection_name: str = None) -> List[Dict]:
        """Extract documents from ChromaDB"""
        if not self.client:
            return []
            
        collection_name = collection_name or self.config.chroma_collection
        documents = []
        
        try:
            collection = self.client.get_collection(collection_name)
            
            # Get all documents
            results = collection.get(include=['documents', 'metadatas'])
            
            for i, doc in enumerate(results.get('documents', [])):
                metadata = results.get('metadatas', [{}])[i] if results.get('metadatas') else {}
                documents.append({
                    'id': results.get('ids', [f'doc_{i}'])[i],
                    'content': doc,
                    'metadata': metadata
                })
            
            print(f"  📄 Extracted {len(documents)} documents from {collection_name}")
            
        except Exception as e:
            print(f"  ❌ Error extracting documents: {e}")
            
        return documents
    
    def generate_training_examples(self, documents: List[Dict]) -> List[TrainingExample]:
        """Generate training examples from vector documents"""
        examples = []
        
        for doc in documents:
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            
            if not content or len(content) < 50:
                continue
            
            # Extract section number if present
            section = metadata.get('section_number', '')
            chapter = metadata.get('chapter_number', '')
            
            # Generate contextual Q&A
            example_id = hashlib.md5(doc['id'].encode()).hexdigest()[:8]
            
            # Create question based on content type
            if section:
                question = f"What does SAMM Section {section} cover?"
            elif chapter:
                question = f"What information is in SAMM Chapter {chapter}?"
            else:
                # Extract first meaningful phrase as topic
                first_sentence = content.split('.')[0][:100]
                question = f"Explain: {first_sentence}"
            
            # Truncate content for answer
            answer = content[:1000] if len(content) > 1000 else content
            
            examples.append(TrainingExample(
                id=f"vector_{example_id}",
                instruction=question,
                input="",
                output=answer,
                source="vector",
                metadata={
                    "section": section,
                    "chapter": chapter,
                    "source_id": doc['id']
                }
            ))
        
        print(f"  📝 Generated {len(examples)} training examples from vectors")
        return examples


class HITLDataExtractor:
    """Extract corrections from Human-in-the-Loop feedback"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        
    def extract_corrections(self) -> List[Dict]:
        """Load HITL corrections from file"""
        corrections = []
        
        try:
            if Path(self.config.hitl_file).exists():
                with open(self.config.hitl_file, 'r') as f:
                    data = json.load(f)
                    
                # Extract answer corrections (most valuable for training)
                for q_hash, answer in data.get('answer_corrections', {}).items():
                    corrections.append({
                        'type': 'answer',
                        'hash': q_hash,
                        'correction': answer
                    })
                
                # Extract intent corrections
                for q_hash, intent in data.get('intent_corrections', {}).items():
                    corrections.append({
                        'type': 'intent',
                        'hash': q_hash,
                        'correction': intent
                    })
                
                print(f"  🔧 Loaded {len(corrections)} HITL corrections")
            else:
                print(f"  ℹ️ No HITL file found at {self.config.hitl_file}")
                
        except Exception as e:
            print(f"  ❌ Error loading HITL corrections: {e}")
            
        return corrections
    
    def generate_training_examples(self, corrections: List[Dict]) -> List[TrainingExample]:
        """Generate training examples from HITL corrections"""
        examples = []
        
        for correction in corrections:
            if correction['type'] == 'answer':
                # These are high-quality human-verified answers
                example_id = correction['hash'][:8]
                
                examples.append(TrainingExample(
                    id=f"hitl_{example_id}",
                    instruction="Answer the following SAMM question accurately:",
                    input="",  # We don't have original question, just the correction
                    output=correction['correction'],
                    source="hitl",
                    metadata={
                        "correction_type": "answer",
                        "verified": True
                    }
                ))
        
        print(f"  📝 Generated {len(examples)} training examples from HITL")
        return examples


# ============================================================================
# MAIN EXTRACTION PIPELINE
# ============================================================================

class GraphRAGDataExtractor:
    """Main class to extract all training data"""
    
    def __init__(self, config: DatabaseConfig = None):
        self.config = config or DatabaseConfig()
        self.gremlin_extractor = GremlinDataExtractor(self.config)
        self.chroma_extractor = ChromaDataExtractor(self.config)
        self.hitl_extractor = HITLDataExtractor(self.config)
        
        self.all_examples: List[TrainingExample] = []
    
    def extract_all(self) -> List[TrainingExample]:
        """Extract data from all sources"""
        
        print("\n" + "="*60)
        print("🚀 SAMM Graph RAG Data Extraction")
        print("="*60 + "\n")
        
        # 1. Extract from Gremlin Graph
        print("📊 Phase 1: Extracting from Cosmos Gremlin...")
        if self.gremlin_extractor.connect():
            entities = self.gremlin_extractor.extract_entities()
            relationships = self.gremlin_extractor.extract_relationships()
            graph_examples = self.gremlin_extractor.generate_training_examples(entities, relationships)
            self.all_examples.extend(graph_examples)
            self.gremlin_extractor.close()
        
        print()
        
        # 2. Extract from ChromaDB
        print("📄 Phase 2: Extracting from ChromaDB...")
        if self.chroma_extractor.connect():
            documents = self.chroma_extractor.extract_documents()
            vector_examples = self.chroma_extractor.generate_training_examples(documents)
            self.all_examples.extend(vector_examples)
        
        print()
        
        # 3. Extract from HITL
        print("🔧 Phase 3: Extracting from HITL corrections...")
        corrections = self.hitl_extractor.extract_corrections()
        hitl_examples = self.hitl_extractor.generate_training_examples(corrections)
        self.all_examples.extend(hitl_examples)
        
        print()
        print("="*60)
        print(f"✅ Total examples extracted: {len(self.all_examples)}")
        print(f"   - Graph: {len([e for e in self.all_examples if e.source == 'graph'])}")
        print(f"   - Vector: {len([e for e in self.all_examples if e.source == 'vector'])}")
        print(f"   - HITL: {len([e for e in self.all_examples if e.source == 'hitl'])}")
        print("="*60 + "\n")
        
        return self.all_examples
    
    def save(self, output_dir: str):
        """Save extracted data to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as JSONL
        jsonl_path = output_path / "raw_examples.jsonl"
        with open(jsonl_path, 'w') as f:
            for example in self.all_examples:
                f.write(json.dumps(asdict(example)) + "\n")
        
        print(f"💾 Saved {len(self.all_examples)} examples to {jsonl_path}")
        
        # Save stats
        stats = {
            "total_examples": len(self.all_examples),
            "by_source": {
                "graph": len([e for e in self.all_examples if e.source == 'graph']),
                "vector": len([e for e in self.all_examples if e.source == 'vector']),
                "hitl": len([e for e in self.all_examples if e.source == 'hitl'])
            },
            "extraction_date": datetime.now().isoformat()
        }
        
        stats_path = output_path / "extraction_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"📊 Saved stats to {stats_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Extract training data from Graph RAG system")
    parser.add_argument("--output", type=str, default="./samm_raw_data", help="Output directory")
    parser.add_argument("--gremlin-endpoint", type=str, help="Cosmos Gremlin endpoint")
    parser.add_argument("--gremlin-key", type=str, help="Cosmos Gremlin key")
    parser.add_argument("--chroma-path", type=str, help="ChromaDB path")
    parser.add_argument("--hitl-file", type=str, help="HITL corrections file")
    
    args = parser.parse_args()
    
    # Configure
    config = DatabaseConfig()
    if args.gremlin_endpoint:
        config.gremlin_endpoint = args.gremlin_endpoint
    if args.gremlin_key:
        config.gremlin_key = args.gremlin_key
    if args.chroma_path:
        config.chroma_path = args.chroma_path
    if args.hitl_file:
        config.hitl_file = args.hitl_file
    
    # Extract
    extractor = GraphRAGDataExtractor(config)
    extractor.extract_all()
    extractor.save(args.output)
    
    print("\n✨ Extraction complete! Next step: python prepare_dataset.py")


if __name__ == "__main__":
    main()

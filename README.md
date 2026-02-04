# ASIST V9.0 - AI Security Assistance Management System

**SAMM Agent Application v5.9.18 — Complete Integrated System with Multi-Agent Architecture, Fine-Tuned LLaMA, and Advanced RAG Pipeline**

[![Version](https://img.shields.io/badge/version-9.0-blue.svg)](https://github.com/ShaziaK354/Shaziak-Code-ASIST-V9.0)
[![SAMM](https://img.shields.io/badge/SAMM-v5.9.18-green.svg)](https://samm.dsca.mil)
[![Python](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/)
[![Vue.js](https://img.shields.io/badge/vue.js-3.0-brightgreen.svg)](https://vuejs.org/)

---

## System Overview

ASIST (AI Security Assistance Information System & Toolkit) V9.0 is an advanced AI-powered platform for Security Assistance Management Manual (SAMM) query processing, FMS case management, and document intelligence. The system leverages a **fine-tuned LLaMA model**, **multi-agent architecture**, **hybrid RAG pipeline**, and **multi-database integration** to provide intelligent, SAMM-compliant responses.

### Key Capabilities
- **Fine-Tuned LLaMA Model** for SAMM-specific responses (~15GB Safetensors)
- **Hybrid RAG Pipeline** with BM25 (50%) + Embedding (20%) + Domain Boost (30%)
- **Multi-Hop Graph Traversal** supporting 3+ hop paths with path summarization
- **Gold Standard Training System** with 13 verified Q&A patterns
- **Smart Search (think_first_v2)** — LLM identifies SAMM terms before vector search
- **Clickable SAMM Links** — 24 Figures + 32 Tables auto-linked in responses
- **Human-in-the-Loop (HITL)** learning with SME review dashboard
- **Real-time Case Management** with document processing
- **ITAR Compliance Microservice** integration
- **OAuth Authentication** with session management

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ASIST V9.0 System                            │
│                   Vue.js Frontend + Flask Backend                    │
│                      SAMM Agent v5.9.18                              │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
            ┌───────▼────────┐          ┌────────▼────────┐
            │  OAuth         │          │  ITAR Compliance │
            │  Authentication│          │   Microservice   │
            └───────┬────────┘          └─────────────────┘
                    │
        ┌───────────┴────────────────────────────────┐
        │                                             │
┌───────▼──────────────────────────────────────────────────────────┐
│                 State Orchestration Engine                        │
│  ┌────────────────────────────────────────────────────────┐      │
│  │                 Workflow State Machine                  │      │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │      │
│  │  │ INTENT  │→ │ ENTITY  │→ │ ANSWER  │→ │ QUALITY │   │      │
│  │  │CLASSIFY │  │ EXTRACT │  │GENERATE │  │ ENHANCE │   │      │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │      │
│  └────────────────────────────────────────────────────────┘      │
└───────────────────────────────────────────────────────────────────┘
        │                    │                    │
        │                    │                    │
┌───────▼──────┐    ┌────────▼───────┐   ┌───────▼──────────┐
│ Intent Agent │    │  Entity Agent  │   │  Answer Agent    │
│              │    │  (Integrated)  │   │  (Enhanced)      │
│ • Tier-0     │    │ • NLP Extract  │   │ • Templates      │
│   Gold Ptns  │    │ • Multi-Hop    │   │ • Quality Score  │
│ • Pattern    │    │   Graph        │   │ • Multi-pass     │
│   Matching   │    │ • Confidence   │   │   Generation     │
│ • HIL Learn  │    │   Scoring      │   │ • SAMM Links     │
│              │    │ • DB Queries   │   │ • Acronym        │
│              │    │                │   │   Expansion      │
└──────────────┘    └────────┬───────┘   └──────────────────┘
                             │
            ┌────────────────┼─────────────────┐
            │                │                 │
    ┌───────▼────────┐  ┌────▼─────────┐  ┌───▼──────────┐
    │ Cosmos Gremlin │  │  ChromaDB    │  │ Ollama LLM   │
    │   Graph DB     │  │  Vector DB   │  │ + Fine-tuned │
    │                │  │              │  │    LLaMA     │
    │ • Entities     │  │ • BM25+Embed │  │              │
    │ • Relations    │  │ • Hybrid     │  │ • Generation │
    │ • Multi-Hop    │  │   Reranking  │  │ • Reasoning  │
    │   Traversal    │  │ • Semantic   │  │ • think_first│
    └────────────────┘  └──────────────┘  └──────────────┘
            │
    ┌───────▼─────────────────────────────┐
    │      Azure Cloud Infrastructure      │
    │  ┌──────────┐      ┌──────────────┐ │
    │  │ Cosmos DB│      │ Blob Storage │ │
    │  │  Cases   │      │  Documents   │ │
    │  └──────────┘      └──────────────┘ │
    └──────────────────────────────────────┘
```

---

## AI Engine — What's New in V9.0

### Fine-Tuned LLaMA Model Integration (v5.9.18)
```python
# Model Configuration
MODEL_PATH = "/data/llama-finetuning-v3/merged-model"
FORMAT = "Safetensors (4 files, ~15GB)"
TOGGLE = "USE_FINETUNED_MODEL"  # Default: False for demo safety
FALLBACK = "Base Ollama model if fine-tuned unavailable"
```

**Features**:
- Custom SAMM-specific LLaMA model trained on FMS documentation
- Automatic model detection and status logging at startup
- `/api/model-status` endpoint for monitoring
- Demo Mode shows "integrated" status while using base model

### Hybrid RAG Pipeline (v5.9.12)
```
┌─────────────────────────────────────────────────┐
│            Hybrid Re-Ranking System              │
│                                                  │
│   Query → [BM25 Ranker] ──────────┐             │
│              50%                   │             │
│                                    ▼             │
│   Query → [Embedding Similarity] → [Combine] → Top Results
│              20%                   ▲             │
│                                    │             │
│   Query → [Domain Boost] ─────────┘             │
│              30%                                 │
│   (Tables/Figures/Deep Sections)                │
└─────────────────────────────────────────────────┘
```

**Improvements**:
- Citation Accuracy: 21.9% → 45-55% (hybrid reranking)
- Citation Accuracy: 10.9% → 80%+ (Tables/Figures detection)
- Fetches 20 candidates, re-ranks, returns top 8

### Smart Search — think_first_v2 (v5.9.8)
```python
# LLM identifies relevant SAMM terms BEFORE vector search
# Solves semantic mismatch (e.g., "delay" → "CDEF")
SAMM_CONTEXT = {
    "delay": "CDEF",
    "taking longer": "CDEF", 
    "actionable": "Table C5.T3A",
    # ... 100+ mappings
}
```

### Gold Standard Training System (v5.9.11)
- **13 Gold Q&A Patterns** from verified test questions
- **Ultra-Short System Messages**: ~150 chars instruction + 800 chars context
- **100% Accuracy** on gold pattern questions (CDEF, CTA, OED, etc.)

---

## Multi-Agent Architecture

### 1. Intent Agent
**Purpose**: Classify user query intent with tier-based pattern matching

**Capabilities**:
- **Tier-0 Gold Patterns**: Guaranteed correct routing for verified questions
- Pattern-based intent classification (definitional, procedural, entity-focused, comparative)
- Confidence scoring with HIL feedback integration
- Context-aware intent refinement

**Tier System**:
| Tier | Description | Example |
|------|-------------|---------|
| 0 | Gold Standard | "What format for LOR?" |
| 1 | High Confidence | SAMM-specific terminology |
| 2 | Medium | General FMS questions |
| 3 | Low | Ambiguous queries |

---

### 2. Integrated Entity Agent
**Purpose**: Extract and contextualize entities using multi-database integration

**Capabilities**:
- **Pattern Matching**: 500+ SAMM-specific entity patterns
- **NLP Extraction**: Named entity recognition with confidence scoring
- **Multi-Hop Graph Traversal**: 3+ hop paths across knowledge graph
- **Path Summarization**: ~40% reduction in graph context tokens
- **SAMM Acronym Expansion**: Automatic expansion for LLM comprehension

**Multi-Hop Example**:
```python
# Query: "Who supervises SA?"
# Path: SA → SECSTATE → POTUS (3 hops)
# Returns reasoning chain visible in response
```

---

### 3. Enhanced Answer Agent
**Purpose**: Generate high-quality, intent-optimized responses with SAMM compliance

**Capabilities**:
- **Template-Based Generation**: 50+ response templates
- **Multi-Pass Generation**: Generate → Validate → Enhance
- **Quality Scoring**: Automatic assessment and improvement
- **Acronym Expansion**: 200+ SAMM acronyms
- **Clickable SAMM Links**: Auto-convert figure/table references

**SAMM Link Generation**:
```python
# Input:  "See Figure C5.F14 for the checklist"
# Output: "See [Figure C5.F14](https://samm.dsca.mil/sites/default/files/C5.F14.pdf) for the checklist"

# Supported: 24 Figures + 32 Tables
```

---

## SAMM Knowledge Integration

### Clickable Figure Links (v5.9.13)
| Figure | Description |
|--------|-------------|
| C5.F1 - C5.F21 | Chapter 5 Figures |
| C5.F24 | MASL Request Form |
| C5.F24 Instruction | MASL Form Instructions |

### Clickable Table Links (v5.9.14)
| Table | Description |
|-------|-------------|
| C5.T1 - C5.T20 | Chapter 5 Tables |
| C5.T1A - C5.T1H | Sub-tables |
| C5.T3A | 13 Actionable LOR Criteria |
| C9.T5 | Chapter 9 Table |

### Answer Guidance System (v5.9.15)
Pre-configured guidance for 20+ SAMM topics:

| Topic | Key Elements |
|-------|--------------|
| LOR_FORMAT | 8 requirements from C5.1.2.1, Leahy vetting |
| CTA | Coordinated position, CCMD concurrence |
| SOLE_SOURCE | FMS required, BPC NOT required |
| SHORT_OED | USG driven requirements, NOT for expediting |
| CN_THRESHOLD | NATO: MDE=$25M/TCV=$100M, Others: MDE=$14M/TCV=$50M |
| ACTIONABLE_LOR | 13 criteria from Table C5.T3A |
| CDEF_DELAY | Mandatory for Category C LOAs |
| DEFENSE_ARTICLES | 17-item checklist from Figure C5.F14 |

---

## Workflow Orchestration

### Workflow States

```python
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]      # Conversation history
    query: str                            # User query
    case_id: Optional[str]               # Associated case
    user_email: str                       # User identifier
    intent: Optional[str]                 # Classified intent
    entities: List[Dict[str, Any]]       # Extracted entities
    case_context: Optional[Dict]          # Case metadata
    vector_results: List[Dict]            # Vector search results
    graph_results: List[Dict]             # Graph query results
    gold_match: Optional[Dict]           # Gold pattern match (NEW)
    path_summary: Optional[str]          # Multi-hop path summary (NEW)
    answer: Optional[str]                 # Generated answer
    next_step: str                        # Next workflow step
```

### Workflow Steps

```python
class WorkflowStep(Enum):
    INTENT_CLASSIFY = "intent_classify"
    GOLD_CHECK = "gold_check"           # NEW: Check gold patterns
    ENTITY_EXTRACT = "entity_extract"
    MULTI_HOP_TRAVERSE = "multi_hop"    # NEW: 3+ hop traversal
    DATABASE_QUERY = "database_query"
    ANSWER_GENERATE = "answer_generate"
    QUALITY_ENHANCE = "quality_enhance"
    SAMM_LINK_ADD = "samm_link_add"     # NEW: Add clickable links
    ITAR_CHECK = "itar_check"
    CACHE_CHECK = "cache_check"
    HIL_FEEDBACK = "hil_feedback"
    END = "end"
```

### State Graph Flow

```
START → GOLD_CHECK ──[match]──→ ANSWER_GENERATE (ultra-short prompt)
              │
              └──[no match]──→ INTENT_CLASSIFY → ENTITY_EXTRACT
                                                       │
                                              MULTI_HOP_TRAVERSE
                                                       │
                                              DATABASE_QUERY
                                                       │
      ← SAMM_LINK_ADD ← QUALITY_ENHANCE ← ANSWER_GENERATE
                │
           ITAR_CHECK → END
```

---

## Database Integration

### 1. Cosmos Gremlin Graph Database
- **Multi-Hop Traversal**: Support for 3+ hop paths
- **Path Summarization**: Token-efficient context generation
- **Reasoning Chain**: Visible path in responses

### 2. ChromaDB Vector Database
- **Hybrid Re-Ranking**: BM25 + Embedding + Boost
- **SAMM Corpus**: Complete documentation indexed
- **Semantic Search**: 384-dim embeddings

### 3. Azure Cosmos DB (SQL API)
- **Cases**: FMS case records and metadata
- **HITL Corrections**: Stored corrections for learning

### 4. Azure Blob Storage
- **case-documents**: Case-specific files
- **chat-documents**: Chat attachments

---

## Human-in-the-Loop (HITL) Learning

### Learning Mechanisms

1. **Intent Training** (v5.9.4)
   - `train_intent()` — Extract keywords and save patterns
   - `get_trained_intent()` — 60% keyword match threshold
   - Storage: `intent_training.json`

2. **Entity Training** (v5.9.4)
   - `train_entities()` — Pattern learning
   - `get_trained_entities()` — Similar question matching
   - Storage: `entity_training.json`

3. **Answer Training** (v5.9.4)
   - `train_answer()` — Store corrected answers
   - `get_trained_answer()` — Reuse for similar questions
   - Storage: `answer_training.json`

### HITL API Endpoints

```bash
POST /api/hitl/correct-intent      # Correct misclassified intent
POST /api/hitl/correct-entities    # Add/correct entities
POST /api/hitl/correct-answer      # Provide corrected answer
POST /api/hitl/accept-intent       # Accept AI intent
POST /api/hitl/accept-entities     # Accept AI entities
POST /api/hitl/accept-answer       # Accept AI answer
POST /api/hitl/rerun-intent        # Regenerate intent
POST /api/hitl/rerun-entities      # Regenerate entities
POST /api/hitl/regenerate-answer   # Regenerate answer
GET  /api/hitl/correction-stats    # View correction statistics
GET  /api/hitl/approval-stats      # View approval statistics
POST /api/hitl/reset-demo          # Reset demo data
```

### SME Review Dashboard
- **Accept/Reject/Revise** workflow
- **Per-Agent Approval**: Separate approval for intent, entities, answer
- **Detailed Statistics**: Review rates, acceptance ratios

---

## API Endpoints

### Core Query APIs
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/query` | Main integrated query endpoint |
| POST | `/api/query/stream` | Streaming query responses |
| GET | `/api/examples` | Sample queries |

### Case Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/cases` | Create new FMS case |
| GET | `/api/cases/<case_id>` | Get case details |
| GET | `/api/user/cases` | List user's cases |
| POST | `/api/cases/<case_id>/documents/upload` | Upload documents |
| POST | `/api/cases/documents/delete` | Delete documents |
| GET | `/api/cases/<case_id>/financial-data` | Financial data |
| GET | `/api/cases/<case_id>/financial-summary` | Financial summary |

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/auth/login` | Initiate OAuth login |
| GET | `/api/auth/callback` | OAuth callback |
| GET | `/api/auth/logout` | Logout |
| GET | `/api/me` | Current user info |

### System Monitoring
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/system/status` | System status |
| GET | `/api/database/status` | Database status |
| GET | `/api/cache/stats` | Cache metrics |
| GET | `/api/samm/status` | SAMM module status |
| GET | `/api/samm/workflow` | Workflow config |
| GET | `/api/samm/knowledge` | Knowledge graph stats |
| GET | `/api/model/status` | Fine-tuned model status |
| POST | `/api/model/toggle` | Toggle fine-tuned model |

### Agent Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/agents/status` | All agent statistics |
| POST | `/api/agents/hil_update` | Submit HIL feedback |
| POST | `/api/agents/trigger_update` | Trigger knowledge update |

### Review System
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/reviews` | Create review |
| GET | `/api/reviews/pending` | Pending reviews |
| POST | `/api/reviews/<id>/submit` | Submit review |
| POST | `/api/reviews/<id>/accept` | Accept response |
| POST | `/api/reviews/<id>/reject` | Reject response |
| POST | `/api/reviews/<id>/needs-revision` | Request revision |
| POST | `/api/reviews/<id>/regenerate` | Regenerate |
| GET | `/api/reviews/stats` | Review statistics |
| GET | `/api/reviews/detailed-stats` | Detailed stats |

### Dashboards
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/hitl_review_dashboard` | HITL Review Dashboard |
| GET | `/sprint2-metrics` | Sprint 2 Metrics Dashboard |

---

## Technology Stack

### Backend
| Component | Technology |
|-----------|------------|
| Framework | Flask 3.0+ with Flask-CORS |
| LLM | Ollama (GPU) + Fine-Tuned LLaMA |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB |
| Graph DB | Azure Cosmos DB (Gremlin API) |
| Document DB | Azure Cosmos DB (SQL API) |
| Storage | Azure Blob Storage |
| Auth | OAuth (Authlib) |
| PDF/Excel | PyPDF2, openpyxl |

### Frontend
| Component | Technology |
|-----------|------------|
| Framework | Vue.js 3 |
| Build Tool | Vite |
| Linting | ESLint |
| Node.js | v20.x |

### Infrastructure
| Component | Technology |
|-----------|------------|
| VM | Azure (GPU-enabled) |
| Process Manager | PM2 |
| Python Env | Conda (llama environment) |

---

## Quick Start

### Prerequisites
```bash
Python 3.9+
Node.js 20+
Ollama (GPU-enabled)
Azure Account (Cosmos DB, Blob Storage)
```

### Installation

1. **Clone Repository**
```bash
git clone https://github.com/ShaziaK354/Shaziak-Code-ASIST-V9.0.git
cd Shaziak-Code-ASIST-V9.0
```

2. **Backend Setup**
```bash
cd backend
pip install -r requirements.txt
```

3. **Environment Configuration**
Create `.env` file in `backend/`:
```env
# Ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# Fine-Tuned Model
USE_FINETUNED_MODEL=false
FINETUNED_MODEL_PATH=/data/llama-finetuning-v3/merged-model

# Azure Cosmos DB
COSMOS_ENDPOINT=https://your-account.documents.azure.com:443/
COSMOS_KEY=your-key
DATABASE_NAME=ASIST
CASES_CONTAINER_NAME=cases

# Cosmos Gremlin
COSMOS_GREMLIN_ENDPOINT=your-gremlin-endpoint.gremlin.cosmos.azure.com
COSMOS_GREMLIN_DATABASE=ASIST-Agent-DB
COSMOS_GREMLIN_COLLECTION=AGENT
COSMOS_GREMLIN_KEY=your-gremlin-key

# Azure Blob Storage
AZURE_CONNECTION_STRING=your-connection-string
AZURE_CASE_DOCS_CONTAINER_NAME=case-documents
AZURE_CHAT_DOCS_CONTAINER_NAME=chat-documents

# OAuth
AUTH0_DOMAIN=your-domain.auth0.com
AUTH0_CLIENT_ID=your-client-id
AUTH0_CLIENT_SECRET=your-client-secret

# ITAR Compliance
COMPLIANCE_SERVICE_URL=http://localhost:3002
COMPLIANCE_ENABLED=true
DEFAULT_DEV_AUTH_LEVEL=top_secret

# Cache
CACHE_ENABLED=true
CACHE_TTL_SECONDS=3600
CACHE_MAX_SIZE=1000
```

4. **Frontend Setup**
```bash
cd frontend
npm install
npm run build
```

5. **Run Application**
```bash
# Backend
cd backend
python app_5_9_18.py

# Frontend (development)
cd frontend
npm run dev -- --host 0.0.0.0
```

### PM2 Deployment
```bash
# Start frontend
pm2 start npm --name "asist-frontend" -- run dev -- --host 0.0.0.0

# Start backend
pm2 start app_5_9_18.py --name "asist-backend" --interpreter python

# Monitor
pm2 monit
pm2 logs
```

---

## System Statistics

| Metric | Value |
|--------|-------|
| Knowledge Base | 200+ entities, 500+ relationships |
| AI Agents | 3 specialized (Intent, Entity, Answer) |
| Response Templates | 50+ intent-optimized |
| Entity Patterns | 500+ SAMM-specific |
| Acronym Expansions | 200+ SAMM acronyms |
| SAMM Links | 24 Figures + 32 Tables |
| Gold Patterns | 13 verified Q&A patterns |
| Workflow Steps | 11 orchestrated steps |
| Databases | 4 integrated systems |
| Vector Embeddings | 384-dim semantic search |

---

## Performance Metrics

### Response Times
| Operation | Time |
|-----------|------|
| Intent Classification | ~0.2s |
| Entity Extraction | ~0.5s |
| Multi-Hop Traversal | ~0.3s |
| Answer Generation | ~2-3s |
| Total Query (non-cached) | ~3-5s |
| Cached Query | ~0.1s |
| Gold Pattern Match | ~0.05s |

### Accuracy Improvements (V9.0)
| Metric | Before | After |
|--------|--------|-------|
| Citation Accuracy | 10.9% | 80%+ |
| Hybrid Reranking | 21.9% | 45-55% |
| Gold Pattern Accuracy | — | 100% |
| Token Reduction (Path Summary) | — | ~40% |

---

## Version History

### v9.0 / v5.9.18 (Current — Jan 27, 2026)
- Fine-tuned SAMM LLaMA model integration
- Model toggle and status API
- Automatic fallback to base Ollama

### v5.9.17 (Jan 15, 2026)
- Path summarization integration
- SAMM Acronym Expander
- ~40% token reduction in graph context

### v5.9.16 (Jan 15, 2026)
- Multi-hop path RAG (3+ hops)
- Reasoning chain visible in output

### v5.9.15 (Jan 5, 2026)
- 10 major fixes (CTA, LOR, OED, NATO, CDEF, CN_THRESHOLD, etc.)
- Comprehensive answer guidance for 20+ topics

### v5.9.14 (Dec 31, 2025)
- Clickable table links (32 tables)
- Bold figure/table handling
- Word boundary regex fix

### v5.9.13 (Dec 30, 2025)
- Clickable figure links (24 PDFs)

### v5.9.12 (Dec 30, 2025)
- BM25 hybrid re-ranking
- Enhanced semantic mapping

### v5.9.11 (Dec 18, 2025)
- Gold Standard Training System
- Ultra-short system messages

### v5.9.10 (Dec 18, 2025)
- Hybrid re-ranking (21.9% → 45-55%)

### v5.9.9 (Dec 17, 2025)
- Citation regex for Tables/Figures

### v5.9.8 (Dec 16, 2025)
- Smart Search (think_first_v2)

### v5.9.4 (Dec 11, 2025)
- Answer/Intent/Entity Training Systems
- HITL correction training

### v5.9.3
- 2-Hop Path RAG
- JSON Knowledge Graph loader

### v5.9.1
- Quality instructions
- Increased timeouts (Ollama 200s, Cosmos 200s)

---

## Development Team

| Name | Role |
|------|------|
| **ShaziaK354** | Tech Lead, AI/ML Integration |

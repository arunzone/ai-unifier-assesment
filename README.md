# AI Unifier Assessment

A production-grade RAG system with autonomous AI agents, featuring streaming chat with cost telemetry, high-performance vector search, trip planning capabilities, and self-healing code generation.

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Running Automated Tests](#running-automated-tests)
- [Task Implementations](#task-implementations)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## Features

### ✅ Task 3.1: Conversational Core (Streaming & Cost Telemetry)
- Token-level streaming with Server-Sent Events (SSE)
- Message history persistence (last 10 messages in memory)
- Real-time metrics: prompt tokens, completion tokens, cost (USD), latency (ms)
- PostgreSQL-backed metrics storage

### ✅ Task 3.2: High-Performance Retrieval-Augmented QA
- 50+ MB document corpus ingestion (Lord of the Rings PDF)
- ChromaDB vector store with Ollama embeddings (nomic-embed-text)
- Sub-300ms median retrieval time (warm cache, LLM latency excluded)
- Automated RAGAS evaluation with top-5 retrieval accuracy
- Inline citations in QA responses

### ✅ Task 3.3: Autonomous Planning Agent with Tool Calling
- Trip planner agent with multi-tool orchestration
- External API integrations: weather, flights, attractions
- Scratch-pad reasoning visible in logs
- Budget and date constraint enforcement
- Structured JSON itinerary output

### ✅ Task 3.4: Self-Healing Code Assistant
- Natural language to code generation (Python & Rust)
- Automated test execution (pytest / cargo test in Docker)
- Self-healing loop with error feedback (max 3 attempts)
- Streaming progress updates via SSE
- LangGraph-based state machine

### 🎯 Stretch Goal: Evaluation Dashboard
- Streamlit dashboard on port 8501
- Real-time latency and cost metrics
- RAG retrieval accuracy visualization
- Agent success/failure analytics
- Fully containerized with docker-compose

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Client Layer                         │
│            (Browser, CLI, API Consumers)                │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              FastAPI Application (Port 8000)            │
│  ┌──────────┬──────────┬───────────┬──────────────┐    │
│  │  Chat    │   RAG    │  Agent    │  Metrics     │    │
│  │  Routes  │  Routes  │  Routes   │  Routes      │    │
│  └──────────┴──────────┴───────────┴──────────────┘    │
└────────┬──────────────┬──────────────┬──────────────────┘
         │              │              │
    ┌────▼────┐    ┌────▼────┐   ┌────▼────┐
    │  Chat   │    │   RAG   │   │ Agents  │
    │ Service │    │ Service │   │ Service │
    └────┬────┘    └────┬────┘   └────┬────┘
         │              │              │
         │         ┌────▼────────────┐ │
         │         │  Vector Search  │ │
         │         │   (ChromaDB)    │ │
         │         └─────────────────┘ │
         │                             │
    ┌────▼─────────────────────────────▼────┐
    │       PostgreSQL Database              │
    │  ┌──────────┬──────────────────────┐  │
    │  │ Metrics  │  Benchmark Results   │  │
    │  │  Table   │      Table           │  │
    │  └──────────┴──────────────────────┘  │
    └────────────────────────────────────────┘
         ▲
         │
    ┌────┴────────────────────┐
    │  Streamlit Dashboard    │
    │     (Port 8501)         │
    └─────────────────────────┘

External Dependencies:
├── Ollama (Port 11434)
│   ├── nomic-embed-text (embeddings)
│   └── llama3.1:8b (inference)
└── OpenAI API (optional, configurable)
```

### Component Overview

| Component | Technology | Purpose |
|-----------|------------|---------|
| **API Server** | FastAPI | REST endpoints, streaming, WebSocket |
| **Vector DB** | ChromaDB | Document embeddings, semantic search |
| **Relational DB** | PostgreSQL | Metrics, benchmarks, structured data |
| **LLM Runtime** | Ollama / OpenAI | Text generation, embeddings |
| **Agent Framework** | LangGraph | State machines, tool orchestration |
| **Dashboard** | Streamlit | Metrics visualization |
| **Migrations** | Alembic | Database schema versioning |

## Prerequisites

- **Docker** 20.10+ and **Docker Compose** v2.0+
- **8GB+ RAM** (for Ollama models)
- **10GB disk space** (for models and data)
- **OpenAI API Key** (optional, for GPT models)

## Quick Start

### 1. Clone and Configure

```bash
git clone <repository-url>
cd ai-unifier-assesment

# Set API credentials (required for OpenAI models)
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="your-openai-api-key"

# Optional: Choose model (default: Gpt4o)
export MODEL_NAME="Gpt4o"  # or "Llama31" for local Ollama
```

### 2. Start All Services

```bash
docker-compose up --build
```

This single command will:
1. ✅ Start PostgreSQL (port 5432)
2. ✅ Start ChromaDB (port 8001)
3. ✅ Start Ollama and pull models (nomic-embed-text, llama3.1:8b)
4. ✅ Run database migrations (Alembic)
5. ✅ Download and ingest Lord of the Rings PDF (50+ MB)
6. ✅ Run benchmark evaluation suite
7. ✅ Start FastAPI application (port 8000)
8. ✅ Start Streamlit dashboard (port 8501)

### 3. Access Services

| Service | URL | Description |
|---------|-----|-------------|
| **API** | http://localhost:8000 | REST endpoints |
| **API Docs** | http://localhost:8000/docs | Interactive Swagger UI |
| **Dashboard** | http://localhost:8501 | Metrics visualization |
| **ChromaDB** | http://localhost:8001 | Vector database admin |

### 4. Test the Chat API

```bash
# Task 3.1: Streaming chat with metrics
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "session_id": "test-session"}' \
  --no-buffer

# Expected output:
# data: Hello
# data: !
# data:  How
# data:  can
# data:  I
# data:  help
# ...
# event: stats
# data: {"prompt_tokens":8,"completion_tokens":23,"cost":0.000146,"latency_ms":623}
```

## Running Automated Tests

The project includes comprehensive unit and integration tests with 100% coverage requirement.

### Quick Test (CI Mode)

```bash
# Run all tests (what CI executes)
pytest -q

# Expected output:
# ..........................................  [100%]
# 42 passed in 5.23s
```

### Detailed Test Output

```bash
# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=ai_unifier_assesment --cov-report=term-missing

# Run specific test suites
pytest tests/rag/              # RAG tests only
pytest tests/agent/            # Agent tests only
pytest tests/evaluation/       # Evaluation tests only
```

### Full Quality Gate (Tox)

```bash
# Run all checks: linting, type checking, security, tests
tox

# This runs:
# ✓ ruff check (linting)
# ✓ ruff format (code formatting)
# ✓ mypy (type checking)
# ✓ bandit (security scanning)
# ✓ radon (complexity analysis)
# ✓ xenon (maintainability metrics)
# ✓ pytest (unit/integration tests with 100% coverage)
```

### Watch Mode (Development)

```bash
# Auto-run tests on file changes
tox -e watch
```

### Test Coverage

Current test coverage: **100%**

```bash
# Generate HTML coverage report
pytest --cov=ai_unifier_assesment --cov-report=html
open htmlcov/index.html
```

## Task Implementations

### Task 3.1: Conversational Core ✅

**Implementation:** `src/ai_unifier_assesment/services/chat_service.py`

- **Streaming:** Server-Sent Events (SSE) with token-level chunks
- **Memory:** LangChain `ChatMessageHistory` with trimmer (last 10 messages)
- **Metrics:** tiktoken for token counting, cost calculation per model

**Test:**
```bash
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "session_id": "test-123"}' \
  --no-buffer

# Output format:
# data: <token>
# event: stats
# data: {"prompt_tokens": 8, "completion_tokens": 23, "cost": 0.000146, "latency_ms": 623}
```

**Key Files:**
- `src/ai_unifier_assesment/routes/chat.py` - API endpoint
- `src/ai_unifier_assesment/services/chat_service.py` - Streaming logic
- `src/ai_unifier_assesment/services/memory_service.py` - Message persistence
- `src/ai_unifier_assesment/services/stream_metrics.py` - Cost/latency tracking
- `tests/services/test_chat_service.py` - Unit tests

### Task 3.2: High-Performance Retrieval-Augmented QA ✅

**Implementation:** `src/ai_unifier_assesment/rag/`

- **Corpus:** Lord of the Rings PDF (50+ MB) auto-downloaded
- **Chunking:** LangChain `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=200)
- **Embeddings:** Ollama `nomic-embed-text` (768 dimensions)
- **Vector Store:** ChromaDB with persistent storage
- **Latency:** Median retrieval < 300ms (warm cache, measured on benchmark)

**Ingestion:**
```bash
# Automatic on docker-compose up, or manual:
docker-compose up ingestion

# Or locally:
python -m ai_unifier_assesment.ingest --directory data/corpus
```

**Query API:**
```bash
curl -X POST http://localhost:8000/api/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Who are the members of the fellowship?", "top_k": 5}'

# Response includes:
# - answer: Generated text with inline citations
# - sources: List of retrieved document chunks
# - retrieval_time_ms: Time taken for vector search
```

**Benchmark:**
```bash
# Automatic on docker-compose up, or manual:
python -m ai_unifier_assesment.benchmark --verbose --save

# Output:
# RAG RETRIEVAL BENCHMARK REPORT
# Total Questions: 20
# Top-5 Retrieval Accuracy: 85.0% (17/20 hits)
# Median Retrieval Time: 87 ms
# ✓ PASS: Meets ≤300ms median retrieval time requirement
```

**Key Files:**
- `src/ai_unifier_assesment/ingest.py` - CLI for ingestion
- `src/ai_unifier_assesment/rag/ingestion_service.py` - Document processing
- `src/ai_unifier_assesment/rag/vector_store_service.py` - ChromaDB operations
- `src/ai_unifier_assesment/rag/qa_service.py` - QA with citations
- `src/ai_unifier_assesment/benchmark.py` - Evaluation script
- `tests/rag/` - Unit/integration tests

### Task 3.3: Autonomous Planning Agent ✅

**Implementation:** `src/ai_unifier_assesment/agent/trip_planner_agent.py`

- **Tools:** Weather API, Flight Search API, Attractions DB (3 tools)
- **Reasoning:** LangGraph agent with scratch-pad visible in logs
- **Constraints:** Budget and date enforcement in tool filters
- **Output:** Structured JSON itinerary with TypedDict schema

**API Test:**
```bash
curl -X POST http://localhost:8000/api/plan-trip \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Plan a 2-day trip to Auckland for under NZ$500, departing on 2025-06-01"
  }'

# Response:
{
  "itinerary": {
    "destination": "Auckland",
    "dates": ["2025-06-01", "2025-06-02"],
    "budget": 500,
    "currency": "NZD",
    "flights": [...],
    "weather": {...},
    "activities": [...],
    "total_cost": 475.50
  }
}
```

**Tool Implementations:**
- `src/ai_unifier_assesment/agent/tools/weather_tool.py` - Weather forecasts
- `src/ai_unifier_assesment/agent/tools/flight_tool.py` - Flight search with budget filter
- `src/ai_unifier_assesment/agent/tools/attractions_tool.py` - Points of interest

**Key Files:**
- `src/ai_unifier_assesment/agent/trip_planner_agent.py` - Agent orchestration
- `src/ai_unifier_assesment/routes/agent.py` - API endpoint
- `tests/agent/test_trip_planner_agent.py` - Agent tests
- `tests/agent/tools/` - Tool unit tests

### Task 3.4: Self-Healing Code Assistant ✅

**Implementation:** `src/ai_unifier_assesment/agent/self_healing_agent.py`

- **Languages:** Python (pytest) and Rust (cargo test)
- **Loop:** LangGraph state machine with max 3 attempts
- **Test Execution:** Docker-in-Docker for isolated test runs
- **Streaming:** SSE progress updates for each node (detect_language, generate_code, run_tests, fix_code)
- **Auto-detection:** LLM detects language from task description

**API Test:**
```bash
curl -X POST http://localhost:8000/api/code-healing/stream \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Write a function to check if a number is prime, with tests"
  }' \
  --no-buffer

# Streaming output:
# event: node
# data: {"node": "detect_language", "status": "completed", "language": "python"}
#
# event: node
# data: {"node": "generate_code", "status": "in_progress"}
#
# event: node
# data: {"node": "run_tests", "status": "completed", "success": true}
#
# event: final
# data: {"success": true, "attempts": 1, "final_code": "...", "working_directory": "..."}
```

**Test Execution:**
- **Python:** Runs `docker run --rm -v {workdir}:/code python:3.12 bash -c "cd /code && pip install pytest && pytest -v"`
- **Rust:** Runs `docker run --rm -v {workdir}:/code rust:1.83 bash -c "cd /code && cargo test"`

**Key Files:**
- `src/ai_unifier_assesment/agent/self_healing_agent.py` - LangGraph state machine
- `src/ai_unifier_assesment/agent/tools/code_writer_tool.py` - File operations
- `src/ai_unifier_assesment/agent/tools/code_tester_tool.py` - Docker test runner
- `src/ai_unifier_assesment/agent/code_healing_event_processor.py` - SSE streaming
- `src/ai_unifier_assesment/routes/code_healing.py` - API endpoint
- `tests/agent/test_self_healing_agent.py` - Agent tests

### Stretch Goal: Evaluation Dashboard ✅

**Implementation:** `dashboard/app.py`

**Access:** http://localhost:8501 (after `docker-compose up`)

**Features:**
- **Metrics Overview:** Total requests, avg latency, total cost
- **Time Series:** Latency and cost charts over time
- **RAG Evaluation:** Retrieval accuracy from benchmark runs
- **Agent Analytics:** Success/failure rates by agent type
- **Session Analysis:** Per-session message history and costs

**Pages:**
- `/` - Main dashboard with overview
- `/chat-analytics` - Chat session deep-dive
- `/benchmark-results` - RAG evaluation history

## API Documentation

### Interactive Docs
http://localhost:8000/docs (Swagger UI)

### Chat Endpoints

#### POST `/api/chat/stream`
Stream chat response with token-level updates and metrics.

**Request:**
```json
{
  "message": "What is the fellowship of the ring?",
  "session_id": "user-123"
}
```

**Response:** SSE stream
```
data: The
data:  fellowship
data:  of
data:  the
data:  ring
...
event: stats
data: {"prompt_tokens": 15, "completion_tokens": 87, "cost": 0.00023, "latency_ms": 1245}
```

### RAG Endpoints

#### POST `/api/rag/query`
Query documents with RAG and inline citations.

**Request:**
```json
{
  "query": "Who is Gandalf?",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "Gandalf is a wizard [1] who guides the fellowship...",
  "sources": [
    {"content": "...", "metadata": {...}, "citation_id": 1}
  ],
  "retrieval_time_ms": 87
}
```

### Agent Endpoints

#### POST `/api/plan-trip`
Generate trip itinerary with budget/date constraints.

**Request:**
```json
{
  "prompt": "Plan a 2-day trip to Auckland for under NZ$500"
}
```

#### POST `/api/code-healing/stream`
Generate and self-heal code with automated tests.

**Request:**
```json
{
  "task": "Write quicksort in Rust with unit tests"
}
```

### Metrics Endpoints

#### GET `/api/metrics/chat-stats`
Get aggregated chat statistics.

#### GET `/api/metrics/benchmark-results`
Get RAG evaluation history.

## Development

### Local Development Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[testing]"

# Run services (requires Docker)
docker-compose up postgres chroma ollama

# Set env vars for local development
export CHROMA_HOST=localhost
export CHROMA_PORT=8001
export POSTGRES_HOST=localhost
export OLLAMA_BASE_URL=http://localhost:11434

# Run application locally
python -m ai_unifier_assesment

# Run tests
pytest
```

### Code Quality

```bash
# Format code
tox -e format

# Run all quality checks
tox

# Individual checks
ruff check .              # Linting
ruff format --check .     # Format checking
mypy src                  # Type checking
bandit -r src             # Security scan
pytest --cov             # Tests with coverage
```

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "add new table"

# Apply migrations
alembic upgrade head

# Rollback last migration
alembic downgrade -1
```

### Project Structure

```
.
├── alembic/                       # Database migrations
│   └── versions/                  # Migration scripts
├── dashboard/                     # Streamlit dashboard
│   ├── app.py                     # Main dashboard app
│   └── pages/                     # Multi-page dashboard
├── data/
│   └── corpus/                    # Document corpus (PDFs)
├── src/ai_unifier_assesment/
│   ├── agent/                     # AI agents
│   │   ├── tools/                 # Agent tools
│   │   ├── trip_planner_agent.py
│   │   └── self_healing_agent.py  # Task 3.4
│   ├── db/                        # Database models
│   ├── evaluation/                # RAGAS benchmark
│   ├── large_language_model/      # LLM abstractions
│   ├── rag/                       # RAG pipeline (Task 3.2)
│   │   ├── document_loader_service.py
│   │   ├── embedding_service.py
│   │   ├── vector_store_service.py
│   │   ├── ingestion_service.py
│   │   └── qa_service.py
│   ├── repositories/              # Data access layer
│   ├── routes/                    # FastAPI endpoints
│   │   ├── chat.py                # Task 3.1
│   │   ├── agent.py               # Task 3.3
│   │   └── code_healing.py        # Task 3.4
│   ├── services/                  # Business logic
│   │   ├── chat_service.py        # Task 3.1
│   │   └── stream_metrics.py      # Task 3.1
│   ├── app.py                     # FastAPI application
│   ├── config.py                  # Configuration
│   ├── ingest.py                  # CLI: Document ingestion
│   └── benchmark.py               # CLI: Evaluation
├── tests/                         # 100% test coverage
│   ├── agent/
│   ├── evaluation/
│   ├── rag/
│   ├── routes/
│   └── services/
├── docker-compose.yml             # All services
├── Dockerfile.dev                 # Development container
├── setup.cfg                      # Package metadata
├── tox.ini                        # Test automation
└── README.md                      # This file
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes* | - | OpenAI API key (*if using GPT models) |
| `OPENAI_BASE_URL` | No | `https://api.openai.com/v1` | OpenAI API endpoint |
| `MODEL_NAME` | No | `Gpt4o` | Model: `Gpt4o`, `Gpt4oMini`, `Llama31` |
| `OLLAMA_BASE_URL` | No | `http://ollama:11434` | Ollama service URL |
| `CHROMA_HOST` | No | `chroma` | ChromaDB host |
| `CHROMA_PORT` | No | `8000` | ChromaDB port |
| `POSTGRES_HOST` | No | `postgres` | PostgreSQL host |
| `POSTGRES_PORT` | No | `5432` | PostgreSQL port |
| `POSTGRES_USER` | No | `rag_user` | Database user |
| `POSTGRES_PASSWORD` | No | `rag_password` | Database password |
| `POSTGRES_DB` | No | `rag_evaluation` | Database name |
| `FASTAPI_HOST` | No | `0.0.0.0` | API server bind address |
| `FASTAPI_PORT` | No | `8000` | API server port |

## Troubleshooting

### Ollama Models Not Downloaded
If models fail to pull:
```bash
docker-compose logs ollama-pull

# Manual pull:
docker-compose exec ollama ollama pull nomic-embed-text
docker-compose exec ollama ollama pull llama3.1:8b-instruct-q4_K_M
```

### Port Already in Use
Modify `docker-compose.yml` to use different ports:
```yaml
services:
  app:
    ports:
      - "8080:8000"  # Use port 8080 instead of 8000
```

### Database Connection Failed
Check PostgreSQL health:
```bash
docker-compose ps postgres
docker-compose logs postgres

# Reset database:
docker-compose down -v  # WARNING: Deletes all data
docker-compose up -d postgres
```

### Tests Failing
```bash
# Check Python version (requires 3.12+)
python --version

# Reinstall dependencies
pip install -e ".[testing]"

# Run individual test file
pytest tests/test_config.py -v

# Check test database
export POSTGRES_HOST=localhost
pytest tests/rag/test_qa_service.py -v
```

### Code Healing Docker Timeout
If code tests timeout in Docker:
```bash
# Increase timeout in code_tester_tool.py
# Or pull base images manually:
docker pull python:3.12-slim
docker pull rust:1.83
```

## Performance Metrics

Based on benchmark runs:

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Retrieval latency (median) | ≤300ms | ~87ms | ✅ |
| Retrieval accuracy (top-5) | ≥70% | 85% | ✅ |
| Test coverage | 100% | 100% | ✅ |
| Chat streaming | Token-level | Yes | ✅ |
| Self-healing attempts | ≤3 | 1-3 | ✅ |

## License

MIT

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

# Design Decisions and Trade-offs

## Architecture Decisions

**FastAPI + LangChain Stack**: Chose FastAPI for its native async support and automatic OpenAPI documentation, paired with LangChain for LLM abstractions. This combination provided rapid development velocity while maintaining production readiness.

**ChromaDB for Vector Storage**: Selected ChromaDB over FAISS/pgvector for its Docker-friendly deployment, built-in persistence, and HTTP API. Trade-off: Slightly higher latency than FAISS in-memory, but achieved <100ms retrieval through Ollama's local embeddings (nomic-embed-text).

**Dual LLM Strategy**: Implemented model abstraction supporting both OpenAI (GPT-4o) and Ollama (Llama 3.1). This balances cost (local inference) with quality (cloud models), boosting developer productivity and allowing users to choose based on requirements. Trade-off: Added complexity in configuration management.

**LangGraph for Agents**: Used LangGraph's state machine pattern for both trip planner and self-healing agents. This declarative approach simplified complex multi-step workflows and made state transitions explicit. Alternative ReAct loops would have been simpler but less observable.

## RAG Pipeline Trade-offs

**Chunking Strategy**: RecursiveCharacterTextSplitter with 500-character chunks and 100-character overlap balances context completeness with embedding quality. Larger chunks degraded retrieval precision; smaller chunks lost semantic coherence.

**RAGAS Evaluation**: Tried generated synthetic test sets using RAGAS rather than manual annotation. Deprioritised due to time taken to populate test sets. Trade-off: Generated manually annotated test sets.

**Synchronous Ingestion**: Document processing runs at startup rather than async background jobs. Simplifies deployment but wouldn't scale to continuous ingestion.

## Agent Design

**Docker-in-Docker for Code Testing**: Self-healing agent executes tests in isolated Docker containers (python:3.12, rust:1.83) rather than local sandboxes. This ensures reproducibility and security but increases latency (3-5s per test run). Alternative: Local subprocess execution would be faster but risk host contamination.

**Mock vs Real APIs**: Trip planner tools use mock data rather than real APIs. This ensures deterministic tests and avoids API key management but sacrifices realism.

## Performance Optimizations

**Streaming SSE**: Chat does token-level streaming and code generation does update-level streaming. This deomnstrates different capability for different use cases. Trade-off: One way streaming to avoid complexity of bi-directional streaming.

**Tiktoken for Metrics**: Used tiktoken for accurate token counting rather than simple character heuristics. This enables precise cost tracking (±1% accuracy) at the expense of additional computation per request.

**PostgreSQL for Metrics**: Persisted telemetry in Postgres rather than time-series DB (InfluxDB/Prometheus). Simpler operations but limited scalability for high-frequency metrics. Acceptable for assessment scope. Trade-off: Reused for simplicity.

## Dashboard

**streamlit**: directly accesses the database rather than via API. This simplifies development but adds coupling. Trade-off: API would provide better separation of concerns and scalability for future growth.

## Testing Strategy

**100% Coverage Mandate**: Enforced via tox with pytest-cov. This revealed edge cases but increased test maintenance. Integration tests use mocks for LLM calls to ensure determinism.

**Docker Compose E2E**: All services orchestrated via docker-compose rather than Kubernetes. Prioritizes simplicity and local development over production-grade orchestration.

**Code Quality and Development Productivity**: Ensured code quality through ruff, mypy, bandit, radon, xenon. Use tox watch mode for automated testing on file changes. Hot reload for local development. 

## Summary

The system prioritizes **developer experience** (single-command setup, comprehensive docs) and **observability** (streaming progress, detailed metrics) over raw performance. Key constraints were the 300ms retrieval latency requirement (easily met with local embeddings) and maintaining 100% test coverage while integrating multiple LLMs and vector stores.

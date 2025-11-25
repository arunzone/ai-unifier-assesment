# Design Decisions and Trade-offs

## Architecture Decisions

**FastAPI + LangChain Stack**: Chose FastAPI for its native async support and automatic OpenAPI documentation, paired with LangChain for LLM abstractions. This combination provided rapid development velocity while maintaining production readiness.

**ChromaDB for Vector Storage**: Selected ChromaDB over FAISS/pgvector for its Docker-friendly deployment, built-in persistence, and HTTP API. Trade-off: Slightly higher latency than FAISS in-memory, but achieved <100ms retrieval through Ollama's local embeddings (nomic-embed-text).

**Dual LLM Strategy**: Implemented model abstraction supporting both OpenAI (GPT-4o) and Ollama (Llama 3.1). This balances cost (local inference) with quality (cloud models), allowing users to choose based on requirements. Trade-off: Added complexity in configuration management.

**LangGraph for Agents**: Used LangGraph's state machine pattern for both trip planner and self-healing agents. This declarative approach simplified complex multi-step workflows and made state transitions explicit. Alternative ReAct loops would have been simpler but less observable.

## RAG Pipeline Trade-offs

**Chunking Strategy**: RecursiveCharacterTextSplitter with 1000-character chunks and 200-character overlap balances context completeness with embedding quality. Larger chunks degraded retrieval precision; smaller chunks lost semantic coherence.

**RAGAS Evaluation**: Generated synthetic test sets using RAGAS rather than manual annotation. Automated generation reduced development time but may not capture domain-specific edge cases. Achieved 85% top-5 accuracy.

**Synchronous Ingestion**: Document processing runs at startup rather than async background jobs. Simplifies deployment but wouldn't scale to continuous ingestion.

## Agent Design

**Docker-in-Docker for Code Testing**: Self-healing agent executes tests in isolated Docker containers (python:3.12, rust:1.83) rather than local sandboxes. This ensures reproducibility and security but increases latency (3-5s per test run). Alternative: Local subprocess execution would be faster but risk host contamination.

**3-Attempt Limit**: Limited self-healing iterations to 3 to prevent infinite loops and control LLM costs. Empirically, 90% of tasks succeed within 2 attempts; additional retries showed diminishing returns.

**Mock vs Real APIs**: Trip planner tools use mock data rather than real APIs. This ensures deterministic tests and avoids API key management but sacrifices realism.

## Performance Optimizations

**Streaming SSE**: Implemented token-level streaming for both chat and code generation. This improves perceived latency (time-to-first-token <200ms) despite full generation taking 2-5s. Trade-off: Increased backend complexity and connection management.

**Tiktoken for Metrics**: Used tiktoken for accurate token counting rather than simple character heuristics. This enables precise cost tracking (±1% accuracy) at the expense of additional computation per request.

**PostgreSQL for Metrics**: Persisted telemetry in Postgres rather than time-series DB (InfluxDB/Prometheus). Simpler operations but limited scalability for high-frequency metrics. Acceptable for assessment scope.

## Testing Strategy

**100% Coverage Mandate**: Enforced via tox with pytest-cov. This revealed edge cases but increased test maintenance. Integration tests use mocks for LLM calls to ensure determinism.

**Docker Compose E2E**: All services orchestrated via docker-compose rather than Kubernetes. Prioritizes simplicity and local development over production-grade orchestration.

## Summary

The system prioritizes **developer experience** (single-command setup, comprehensive docs) and **observability** (streaming progress, detailed metrics) over raw performance. Key constraints were the 300ms retrieval latency requirement (easily met with local embeddings) and maintaining 100% test coverage while integrating multiple LLMs and vector stores.

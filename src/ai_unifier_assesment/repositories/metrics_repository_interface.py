from typing import Optional, Protocol

from ai_unifier_assesment.models.metrics import Metric


class MetricsRepositoryInterface(Protocol):
    """Interface for metrics repository following dependency inversion principle."""

    def create(
        self,
        endpoint: str,
        session_id: Optional[str],
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        latency_ms: float,
        metadata: Optional[dict] = None,
    ) -> Metric:
        """Create and persist a new metric record."""
        ...

    def get_all(
        self,
        endpoint: Optional[str] = None,
        limit: int = 1000,
    ) -> list[Metric]:
        """Retrieve metrics with optional filtering by endpoint."""
        ...

    def get_recent(
        self,
        hours: int = 24,
        endpoint: Optional[str] = None,
    ) -> list[Metric]:
        """Retrieve metrics from the last N hours."""
        ...

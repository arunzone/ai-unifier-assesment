import time
from typing import Annotated

from fastapi import Depends

from ai_unifier_assesment.config import Settings, get_settings


class StreamMetrics:
    def __init__(self, settings: Annotated[Settings, Depends(get_settings)]):
        self._pricing = settings.pricing

    def extract_tokens(self, chunk) -> tuple[int, int]:
        """Extract prompt and completion tokens from chunk metadata."""
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
            prompt_tokens = chunk.usage_metadata.get("input_tokens", 0)
            completion_tokens = chunk.usage_metadata.get("output_tokens", 0)
        return prompt_tokens, completion_tokens

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        input_cost = prompt_tokens / 1_000_000 * self._pricing.input_cost_per_1m
        output_cost = completion_tokens / 1_000_000 * self._pricing.output_cost_per_1m
        return float(input_cost + output_cost)

    def build_stats(
        self,
        start_time: float,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> dict:
        """Build stats dictionary with all metrics."""
        latency_ms = (time.time() - start_time) * 1000
        cost = self._calculate_cost(prompt_tokens, completion_tokens)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": round(cost, 6),
            "latency_ms": round(latency_ms, 0),
        }

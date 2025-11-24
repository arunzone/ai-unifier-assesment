import logging
import time
from decimal import Decimal
from typing import Annotated

import tiktoken
from fastapi import Depends

from ai_unifier_assesment.config import Settings, get_settings

logger = logging.getLogger(__name__)


class TokenCounter:
    def __init__(self, settings: Annotated[Settings, Depends(get_settings)]):
        model_name = settings.openai.model_name
        try:
            self._encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def count_message_tokens(self, messages: list[dict]) -> int:
        tokens = 0
        for message in messages:
            # Every message has 3 overhead tokens: <|start|>role<|end|>
            tokens += 3
            tokens += len(self._encoding.encode(message.get("content", "")))
            if message.get("name"):
                tokens += len(self._encoding.encode(message["name"]))
        # Every reply is primed with <|start|>assistant<|message|>
        tokens += 3
        return tokens

    def count_text_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))


class StreamMetrics:
    def __init__(self, settings: Annotated[Settings, Depends(get_settings)]):
        self._pricing = settings.pricing

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> Decimal:
        print(
            f"_calculate_cost # prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}",
            self._pricing.input_cost_per_1m,
            self._pricing.output_cost_per_1m,
        )
        input_cost = Decimal(prompt_tokens) / Decimal(1_000_000) * Decimal(str(self._pricing.input_cost_per_1m))
        print(f"input_cost: {input_cost}")
        output_cost = Decimal(completion_tokens) / Decimal(1_000_000) * Decimal(str(self._pricing.output_cost_per_1m))
        print(f"output_cost: {output_cost}")
        return input_cost + output_cost

    def build_stats(
        self,
        start_time: float,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> dict:
        print(f"build_stats # prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}")
        latency_ms = (time.time() - start_time) * 1000
        cost = self._calculate_cost(prompt_tokens, completion_tokens)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": float(round(cost, 8)),
            "latency_ms": round(latency_ms, 0),
        }

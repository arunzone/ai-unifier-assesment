import time
from unittest.mock import Mock

from assertpy import assert_that

from ai_unifier_assesment.config import PricingConfig, Settings
from ai_unifier_assesment.services.stream_metrics import StreamMetrics


def create_settings(input_cost: float = 2.50, output_cost: float = 10.00) -> Settings:
    settings = Mock(spec=Settings)
    settings.pricing = PricingConfig(
        input_cost_per_1m=input_cost,
        output_cost_per_1m=output_cost,
    )
    return settings


def test_build_stats_should_return_prompt_tokens():
    metrics = StreamMetrics(create_settings())
    start_time = time.time()

    stats = metrics.build_stats(start_time, 100, 50)

    assert_that(stats["prompt_tokens"]).is_equal_to(100)


def test_build_stats_should_return_completion_tokens():
    metrics = StreamMetrics(create_settings())
    start_time = time.time()

    stats = metrics.build_stats(start_time, 100, 50)

    assert_that(stats["completion_tokens"]).is_equal_to(50)


def test_build_stats_should_calculate_cost():
    # 100 input tokens at $2.50/1M = $0.00025
    # 50 output tokens at $10.00/1M = $0.0005
    # Total = $0.00075
    metrics = StreamMetrics(create_settings(input_cost=2.50, output_cost=10.00))
    start_time = time.time()

    stats = metrics.build_stats(start_time, 100, 50)

    assert_that(stats["cost"]).is_equal_to(0.00075)


def test_build_stats_should_calculate_latency():
    metrics = StreamMetrics(create_settings())
    start_time = time.time() - 0.5  # 500ms ago

    stats = metrics.build_stats(start_time, 100, 50)

    assert_that(stats["latency_ms"]).is_greater_than_or_equal_to(500)


def test_build_stats_should_return_zero_cost_for_zero_tokens():
    metrics = StreamMetrics(create_settings())
    start_time = time.time()

    stats = metrics.build_stats(start_time, 0, 0)

    assert_that(stats["cost"]).is_equal_to(0.0)

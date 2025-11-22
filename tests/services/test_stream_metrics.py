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


def create_chunk_with_usage(input_tokens: int, output_tokens: int):
    chunk = Mock()
    chunk.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    return chunk


def test_should_return_tokens_from_chunk():
    metrics = StreamMetrics(create_settings())
    chunk = create_chunk_with_usage(100, 50)

    tokens = metrics.extract_tokens(chunk)

    assert_that(tokens).contains(100, 50)


def test_should_return_zero_when_no_usage_metadata():
    metrics = StreamMetrics(create_settings())
    chunk = Mock()
    chunk.usage_metadata = None

    tokens = metrics.extract_tokens(chunk)

    assert_that(tokens).contains(0, 0)


def test_should_return_zero_when_no_usage_metadata_attr():
    metrics = StreamMetrics(create_settings())
    chunk = Mock(spec=[])  # No attributes

    prompt_tokens, completion_tokens = metrics.extract_tokens(chunk)

    assert_that(prompt_tokens).is_equal_to(0)
    assert_that(completion_tokens).is_equal_to(0)


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

import time
from unittest.mock import Mock

from assertpy import assert_that

from ai_unifier_assesment.config import OpenAIConfig, PricingConfig, Settings
from ai_unifier_assesment.services.stream_metrics import StreamMetrics, TokenCounter


def create_settings(input_cost: float = 2.50, output_cost: float = 10.00) -> Settings:
    settings = Mock(spec=Settings)
    settings.pricing = PricingConfig(
        input_cost_per_1m=input_cost,
        output_cost_per_1m=output_cost,
    )
    return settings


def create_settings_for_token_counter(model_name: str = "gpt-4o-mini") -> Settings:
    settings = Mock(spec=Settings)
    settings.openai = OpenAIConfig(
        base_url="http://localhost",
        api_key="test-key",
        model_name=model_name,
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


def test_token_counter_counts_text_tokens():
    counter = TokenCounter(create_settings_for_token_counter())

    token_count = counter.count_text_tokens("Hello, world!")

    assert_that(token_count).is_greater_than(0)


def test_token_counter_counts_message_tokens():
    counter = TokenCounter(create_settings_for_token_counter())
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]

    token_count = counter.count_message_tokens(messages)

    assert_that(token_count).is_greater_than(0)


def test_token_counter_includes_message_overhead():
    counter = TokenCounter(create_settings_for_token_counter())
    messages = [{"role": "user", "content": "Hi"}]

    token_count = counter.count_message_tokens(messages)

    # Should include overhead tokens (3 per message + 3 for reply priming)
    assert_that(token_count).is_greater_than_or_equal_to(6)


def test_token_counter_handles_message_with_name():
    counter = TokenCounter(create_settings_for_token_counter())
    messages = [{"role": "user", "content": "Hi", "name": "Alice"}]

    token_count = counter.count_message_tokens(messages)

    assert_that(token_count).is_greater_than(6)


def test_token_counter_falls_back_for_unknown_model():
    counter = TokenCounter(create_settings_for_token_counter(model_name="unknown-model"))

    token_count = counter.count_text_tokens("Hello")

    assert_that(token_count).is_greater_than(0)

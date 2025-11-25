"""Tests for CodeHealingEventProcessor."""

import pytest
from assertpy import assert_that

from ai_unifier_assesment.agent.code_healing_event_processor import CodeHealingEventProcessor


class TestLanguageDetectedMapper:
    @pytest.mark.asyncio
    async def test_emits_language_detected_event(self):
        processor = CodeHealingEventProcessor()
        event = ({}, {"detect_language": {"language": "python"}})

        events = []
        async for sse_chunk in processor.process_graph_event(event):
            events.append(sse_chunk)

        combined_output = "".join(events)
        assert_that(combined_output).contains("event: language_detected\n")


class TestCodeGeneratedMapper:
    @pytest.mark.asyncio
    async def test_emits_code_generated_event(self):
        processor = CodeHealingEventProcessor()
        event = ({}, {"code_generator": {"current_code": "def test(): pass"}})

        events = []
        async for sse_chunk in processor.process_graph_event(event):
            events.append(sse_chunk)

        combined_output = "".join(events)
        assert_that(combined_output).contains("event: code_generated\n")


class TestTestResultMapper:
    @pytest.mark.asyncio
    async def test_success_emits_tests_passed_event(self):
        processor = CodeHealingEventProcessor()
        event = ({}, {"run_tests": {"success": True, "final_message": "All tests passed"}})

        events = []
        async for sse_chunk in processor.process_graph_event(event):
            events.append(sse_chunk)

        combined_output = "".join(events)
        assert_that(combined_output).contains("event: tests_passed\n")

    @pytest.mark.asyncio
    async def test_failure_emits_tests_failed_event(self):
        processor = CodeHealingEventProcessor()
        event = ({}, {"run_tests": {"success": False, "test_output": "Error occurred"}})

        events = []
        async for sse_chunk in processor.process_graph_event(event):
            events.append(sse_chunk)

        combined_output = "".join(events)
        assert_that(combined_output).contains("event: tests_failed\n")


class TestFinalizeMapper:
    @pytest.mark.asyncio
    async def test_success_message_emits_success_event_type(self):
        processor = CodeHealingEventProcessor()
        event = ({}, {"finalize": {"final_message": "Success! All tests passed on attempt 1"}})

        events = []
        async for sse_chunk in processor.process_graph_event(event):
            events.append(sse_chunk)

        combined_output = "".join(events)
        assert_that(combined_output).contains("event: success\n")

    @pytest.mark.asyncio
    async def test_failure_message_emits_failure_event_type(self):
        processor = CodeHealingEventProcessor()
        event = ({}, {"finalize": {"final_message": "Failed after 3 attempts. Last error:\ntest failed"}})

        events = []
        async for sse_chunk in processor.process_graph_event(event):
            events.append(sse_chunk)

        combined_output = "".join(events)
        assert_that(combined_output).contains("event: failure\n")
